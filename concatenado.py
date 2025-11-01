#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
detect_line_weapons_gst.py — Ambulantes (linha + ByteTrack) + Armas (YOLO) + GStreamer + S3/SQS

✔ Lê linhas de lines.json (A/B/C/...).
✔ YOLOv8: pessoas (0) + armas (43=faca, 44=tesoura) no mesmo frame.
✔ Rastreamento de pessoas com ByteTrack "segura-ID" (~4s, cfg gerado).
✔ Ponto de detecção = PÉS (centro da base do bbox, com leve offset).
✔ Padrões múltiplos: A→B→A, B→C→B, A→B→C→B→A, ...
✔ Grava roll .ts (GStreamer splitmuxsink) → concatena clipe ±N s (ffmpeg) → sobe ao S3.
✔ **Só envia ao SQS se o CLIP foi confirmado no S3** (sem duplicidade).
"""

import os, sys, time, glob, subprocess, tempfile, json, hashlib, traceback, uuid, math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

import numpy as np
import cv2

from ultralytics import YOLO

# ByteTrack (Ultralytics)
try:
    # Ultralytics >=8.0 estrutura:
    from ultralytics.trackers.byte_tracker import BYTETracker, STrack
except Exception as e:
    print("[ERRO] Falhou import do BYTETracker. Atualize ultralytics: pip install -U ultralytics")
    raise

# ================== CONFIG (env sobrescreve) ==================
LINES_JSON    = os.getenv("LINES_JSON", "lines.json")

S3_BUCKET      = os.getenv("S3_BUCKET", "590183969767-detected-images")
S3_PREFIX      = os.getenv("S3_PREFIX", "weapons/")
AWS_REGION     = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
SQS_QUEUE_URL  = os.getenv("SQS_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/590183969767/detected-images")
SQS_IS_FIFO    = SQS_QUEUE_URL.endswith(".fifo")
SQS_GROUP      = os.getenv("SQS_GROUP", "camera-1")
CAMERA_ID      = os.getenv("CAMERA_ID", "train-camera-01")

ROLL_DIR       = os.getenv("ROLL_DIR", "/var/tmp/roll")
SAVE_DIR       = os.getenv("SAVE_DIR", "/home/ubuntu/weapon_events")

PRE            = float(os.getenv("CLIP_PRE",  "5"))    # s antes
POST           = float(os.getenv("CLIP_POST", "5"))    # s depois
CONF           = float(os.getenv("CONF", "0.35"))
IOU            = float(os.getenv("IOU",  "0.45"))
DEVICE         = os.getenv("DEVICE", "cuda")           # "cuda" ou "cpu"
FRAME_STRIDE   = int(os.getenv("FRAME_STRIDE", "2"))   # processa 1 a cada N frames

RTP_PORT       = int(os.getenv("RTP_PORT", "5004"))
PAYLOAD_PT     = int(os.getenv("PAYLOAD_PT", "98"))
CODEC_NAME     = os.getenv("CODEC_NAME", "H265")       # "H265" ou "H264"
LATENCY_MS     = int(os.getenv("LATENCY_MS", "120"))
COOLDOWN       = float(os.getenv("EVENT_COOLDOWN", "8.0"))

# ByteTrack "segura-ID"
TRACKER_CFG          = os.getenv("TRACKER_CFG", "bytetrack.yaml")
HOLD_ID_SECONDS      = float(os.getenv("HOLD_ID_SECONDS", "4.0"))
DEFAULT_FPS_FALLBACK = float(os.getenv("DEFAULT_FPS_FALLBACK", "30.0"))

# Ponto do pé
FOOT_OFFSET_PIXELS = int(os.getenv("FOOT_OFFSET_PIXELS", "5"))
FOOT_PERCENT       = float(os.getenv("FOOT_PERCENT", "0.05"))
CROSS_MARGIN       = float(os.getenv("CROSS_MARGIN", "1.0"))
DRAW_FOOT_POINTS   = os.getenv("DRAW_FOOT_POINTS", "1") in ("1","true","yes")

# Padrões aceitos
ALLOWED_PATTERNS = [
    ["A", "B", "A"],
    ["B", "A", "B"],
    ["A", "C", "A"],
    ["B", "C", "B"],
    ["A", "B", "C", "B", "A"],
    ["B", "A", "C", "A", "B"],
]
SEQ_WINDOW_SEC = float(os.getenv("SEQ_WINDOW_SEC", "60.0"))
RETRIGGER_SEC  = float(os.getenv("RETRIGGER_SEC",  "5.0"))

# ================== AWS ==================
import boto3
s3  = boto3.client("s3",  region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)

def log_err(prefix, e):
    print(prefix, "=>", repr(e))
    traceback.print_exc()

def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(ROLL_DIR, exist_ok=True)
    # teste de escrita
    testfile = os.path.join(ROLL_DIR, ".wtest")
    open(testfile, "w").close()
    os.remove(testfile)

def upload_and_confirm(local_path, key):
    try:
        print(f"[S3] upload {local_path} -> s3://{S3_BUCKET}/{key}")
        s3.upload_file(local_path, S3_BUCKET, key)
        head = s3.head_object(Bucket=S3_BUCKET, Key=key)
        meta = {
            "s3_uri": f"s3://{S3_BUCKET}/{key}",
            "etag": head.get("ETag","").strip('"'),
            "size": head.get("ContentLength", 0),
            "last_modified": head.get("LastModified").isoformat() if head.get("LastModified") else None,
            "version_id": head.get("VersionId")
        }
        print("[S3] confirmado:", meta)
        return meta
    except Exception as e:
        log_err("[S3] falha", e)
        return None

def send_sqs(payload: dict):
    try:
        body = json.dumps(payload, ensure_ascii=False)
        params = {"QueueUrl": SQS_QUEUE_URL, "MessageBody": body}
        if SQS_IS_FIFO:
            dedup = hashlib.md5(body.encode("utf-8")).hexdigest()
            params.update({"MessageGroupId": SQS_GROUP, "MessageDeduplicationId": dedup})
        resp = sqs.send_message(**params)
        print("[SQS] enviado id:", resp.get("MessageId"))
    except Exception as e:
        log_err("[SQS] falha", e)

def pick_segments_by_mtime(center_ts, pre_s=5.0, post_s=5.0):
    start, end = center_ts - pre_s, center_ts + post_s
    files = sorted(glob.glob(os.path.join(ROLL_DIR, "seg-*.ts")))
    chosen = []
    for f in files:
        try:
            m = os.path.getmtime(f)
            if (m >= start - 1.0) and (m <= end + 1.0):
                chosen.append((m, f))
        except FileNotFoundError:
            pass
    chosen.sort()
    return [f for _, f in chosen]

def concat_ts_to_mp4(files, out_mp4):
    if not files:
        print("[CLIP] sem arquivos .ts")
        return False
    with tempfile.NamedTemporaryFile("w", delete=False) as lst:
        for f in files:
            lst.write(f"file '{os.path.abspath(f)}'\n")
        list_path = lst.name
    try:
        cmd = ["ffmpeg","-y","-loglevel","error","-f","concat","-safe","0",
               "-i", list_path, "-c","copy", out_mp4]
        print("[FFMPEG]", " ".join(cmd))
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        log_err("[FFMPEG] erro", e)
        return False
    finally:
        try: os.unlink(list_path)
        except: pass

# =============== Linhas / geometria ===============
def load_lines(path):
    if not os.path.exists(path):
        print(f"[ERRO] {path} não encontrado. Gere com mark_line.py")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    default_names = ["A","B","C"]
    for i, d in enumerate(data):
        if "name" not in d:
            d["name"] = default_names[i] if i < len(default_names) else f"L{i+1}"
    print(f"[LINES] {len(data)} linha(s) carregadas")
    return data

def side_of_line(p, a, b):
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

def crossed(prev, curr, a, b, margin=0.0):
    s1 = side_of_line(prev, a, b)
    s2 = side_of_line(curr, a, b)
    if abs(s1) <= margin or abs(s2) <= margin:
        return None
    if s1 * s2 < 0:
        return 1 if s2 > 0 else -1
    return None

def draw_line_with_name(frame, x1,y1,x2,y2, name):
    cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    mx, my = (x1+x2)//2, (y1+y2)//2
    cv2.putText(frame, name, (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# =============== Sequência de padrões ===============
class SequenceMonitor:
    def __init__(self, patterns: List[List[str]], window_sec: float, retrigger_sec: float):
        self.patterns = [tuple(p) for p in patterns]
        self.window_sec = window_sec
        self.retrigger_sec = retrigger_sec
        self.by_track = defaultdict(lambda: deque(maxlen=32))
        self.last_fire = {}

    def add(self, tid: int, label: str, t: float) -> Optional[Tuple[str, ...]]:
        q = self.by_track[tid]
        if q and q[-1][0] == label:
            return None
        q.append((label, t))
        # janela deslizante
        while q and (t - q[0][1] > self.window_sec):
            q.popleft()
        seq = [x[0] for x in q]
        for pat in self.patterns:
            L = len(pat)
            if len(seq) >= L and tuple(seq[-L:]) == pat:
                if t - self.last_fire.get(tid, 0) >= self.retrigger_sec:
                    self.last_fire[tid] = t
                    return pat
        return None

# =============== ByteTrack auxiliar ===============
class ByteTrackWrapper:
    """
    Envolve BYTETracker para receber detecções (pessoas) como [x1,y1,x2,y2,score]
    e devolver tracks com IDs e bboxes.
    """
    def __init__(self, frame_rate: float, hold_id_seconds: float, cfg_path: str):
        self.frame_rate = int(round(frame_rate)) if frame_rate > 0 else int(DEFAULT_FPS_FALLBACK)
        self.track_buffer = max(30, min(int(round(self.frame_rate * hold_id_seconds)), 300))
        self.tracker = BYTETracker(self._build_args(), frame_rate=self.frame_rate)
        self.frame_id = 0
        self._ensure_cfg_file(cfg_path)

    def _build_args(self):
        # Valores “segura-ID”
        class Args:
            track_high_thresh = 0.5
            track_low_thresh  = 0.1
            new_track_thresh  = 0.4
            match_thresh      = 0.8
            fuse_score        = True
            mot20             = False
            track_buffer      = None
        a = Args()
        a.track_buffer = self.track_buffer
        return a

    def _ensure_cfg_file(self, path: str):
        if os.path.exists(path):
            print(f"[TRACKER] Usando cfg existente: {path}")
            return
        cfg = f"""# {os.path.basename(path)} — gerado automaticamente (~{HOLD_ID_SECONDS:.1f}s)
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.4
track_buffer: {self.track_buffer}
match_thresh: 0.8
fuse_score: True
mot20: False
frame_rate: {self.frame_rate}
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(cfg)
        print(f"[TRACKER] Gerado {path} (frame_rate={self.frame_rate}, track_buffer={self.track_buffer})")

    def update(self, person_dets: np.ndarray, img_wh: Tuple[int,int]):
        """
        person_dets: ndarray Nx5 -> [x1,y1,x2,y2,score] (apenas pessoas)
        img_wh: (w,h) para normalização interna
        """
        self.frame_id += 1
        if person_dets is None or len(person_dets) == 0:
            person_dets = np.zeros((0,5), dtype=float)
        # BYTETracker espera dets no formato xyxy+score
        online_targets: List[STrack] = self.tracker.update(person_dets, [img_wh[0], img_wh[1]], [img_wh[0], img_wh[1]])
        tracks = []
        for t in online_targets:
            tlwh = t.tlwh
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            tid = int(t.track_id)
            tracks.append((tid, int(x1), int(y1), int(x2), int(y2)))
        return tracks

# =============== GStreamer ============
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)

def build_pipeline():
    if CODEC_NAME.upper() == "H265":
        depay_parse  = "rtph265depay ! h265parse config-interval=1 ! "
        decode_branch = "avdec_h265"
        parse_to_roll = "h265parse"
    else:
        depay_parse  = "rtph264depay ! h264parse config-interval=1 ! "
        decode_branch = "avdec_h264"
        parse_to_roll = "h264parse"
    return (
        f'udpsrc port={RTP_PORT} caps=application/x-rtp,media=video,encoding-name={CODEC_NAME},payload={PAYLOAD_PT},clock-rate=90000 ! '
        f'rtpjitterbuffer latency={LATENCY_MS} drop-on-latency=true do-lost=true ! '
        f'{depay_parse}'
        'tee name=t '
        # IA
        f't. ! queue leaky=2 max-size-buffers=30 ! {decode_branch} ! videoconvert ! video/x-raw,format=BGR ! '
        'appsink name=aisink emit-signals=true max-buffers=2 drop=true sync=false '
        # Rolling .ts
        f't. ! queue ! {parse_to_roll} ! '
        f'splitmuxsink name=rollsink muxer=mpegtsmux location="{ROLL_DIR}/seg-%010d.ts" max-size-time=1000000000 max-files=300 '
    )

# =============== Detector principal ===============
class Detector:
    def __init__(self):
        self.lines = load_lines(LINES_JSON)
        if not self.lines:
            print("[ERRO] sem linhas — encerra.")
            sys.exit(1)

        # Pipeline
        self.pipeline  = Gst.parse_launch(build_pipeline())
        self.aisink    = self.pipeline.get_by_name("aisink")
        if not self.aisink:
            print("ERRO: não achou appsink 'aisink'.")
            sys.exit(1)
        self.aisink.connect("new-sample", self.on_new_sample)

        # YOLO único p/ pessoas+armas
        self.model = YOLO("yolov8s.pt")  # pode trocar por seu peso custom
        try:
            import torch
            if DEVICE == "cpu":
                torch.set_num_threads(max(1, os.cpu_count() // 2))
            print(f"[INFO] Torch CUDA? {torch.cuda.is_available()}")
        except Exception:
            pass

        # ByteTrack wrapper
        fps_est = DEFAULT_FPS_FALLBACK  # fps do appsink pode variar; usamos fallback
        self.bt = ByteTrackWrapper(fps_est, HOLD_ID_SECONDS, TRACKER_CFG)

        # Estados
        self.prev_foot: Dict[int, Tuple[int,int]] = {}
        self.seqmon = SequenceMonitor(ALLOWED_PATTERNS, SEQ_WINDOW_SEC, RETRIGGER_SEC)
        self.last_weapon_event = 0.0
        self.last_ambulant_event = 0.0

        self.cnt = 0
        self.t0  = time.time()

    def run(self):
        try:
            ensure_dirs()
        except Exception as e:
            log_err("[DIR] erro", e); sys.exit(1)

        self.pipeline.set_state(Gst.State.PLAYING)
        print(f"[GST] PLAYING ({CODEC_NAME} @ RTP {RTP_PORT}, PT={PAYLOAD_PT})")
        bus = self.pipeline.get_bus()
        try:
            while True:
                msg = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if msg and msg.type == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    print("[GST][ERR]", err, dbg)
                    break
                if msg and msg.type == Gst.MessageType.EOS:
                    print("[GST] EOS")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            print("[GST] stopped")

    # ------- Helpers -------
    def _feet_point(self, x1,y1,x2,y2):
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        cx = (x1 + x2) // 2
        cy_offset  = y2 - FOOT_OFFSET_PIXELS
        cy_percent = y2 - int(h * FOOT_PERCENT)
        cy = max(y1, min(y2, int((cy_offset + cy_percent) / 2)))
        return (int(cx), int(cy))

    def _notify_clip(self, ts_now: float, kind: str, frame: np.ndarray, extra: dict):
        # Snapshot
        snaps_dir = os.path.join(SAVE_DIR, "snapshots"); os.makedirs(snaps_dir, exist_ok=True)
        snap_name = f"{kind}-{int(ts_now)}.jpg"
        snap_path = os.path.join(snaps_dir, snap_name)
        key_snap = None; meta_snap = None
        try:
            cv2.imwrite(snap_path, frame)
            key_snap = f"{S3_PREFIX}snapshots/{snap_name}"
            meta_snap = upload_and_confirm(snap_path, key_snap)
        except Exception as e:
            log_err("[SNAP] falha", e)

        # Clip
        time.sleep(POST + 0.7)  # garante fechamento dos .ts “depois”
        segs = pick_segments_by_mtime(ts_now, PRE, POST)
        clips_dir = os.path.join(SAVE_DIR, "clips"); os.makedirs(clips_dir, exist_ok=True)
        clip_name = f"{kind}-{int(ts_now)}.mp4"
        out_mp4   = os.path.join(clips_dir, clip_name)
        ok_clip = concat_ts_to_mp4(segs, out_mp4)

        meta_clip = None; key_clip = None
        if ok_clip:
            key_clip = f"{S3_PREFIX}clips/{clip_name}"
            meta_clip = upload_and_confirm(out_mp4, key_clip)

        if meta_clip is not None:
            payload = {
                "event": "weapon_detected" if kind=="weapon" else "ambulant_line_cross",
                "camera_id": CAMERA_ID,
                "bucket": S3_BUCKET,
                "ts_epoch": int(ts_now),
                "window": {"pre_s": PRE, "post_s": POST},
                "codec": CODEC_NAME,
                "kind": "clip",
                "s3_key": key_clip,
                "metadata": meta_clip,
            }
            for k,v in (extra or {}).items():
                payload[k] = v
            if meta_snap is not None and key_snap is not None:
                payload["snapshot"] = {"s3_key": key_snap, "metadata": meta_snap}
            send_sqs(payload)
        else:
            print("[SQS] NÃO enviado: clipe não confirmado no S3.")

    # ------- GStreamer callback -------
    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf  = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = int(caps.get_value('width')), int(caps.get_value('height'))
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
        finally:
            buf.unmap(mapinfo)

        self.cnt += 1
        if FRAME_STRIDE > 1 and (self.cnt % FRAME_STRIDE) != 0:
            return Gst.FlowReturn.OK

        # YOLO único com classes [0,43,44]
        res = self.model.predict(
            frame, imgsz=640, conf=CONF, iou=IOU, classes=[0,43,44], device=DEVICE, verbose=False
        )[0]

        people_dets = []   # para ByteTrack: [x1,y1,x2,y2,score]
        weapons = []       # (x1,y1,x2,y2,cls)

        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                cls = int(b.cls.item())
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                score = float(b.conf.item())
                if cls == 0:
                    people_dets.append([x1,y1,x2,y2,score])
                elif cls in (43,44):
                    weapons.append((x1,y1,x2,y2,cls))

        # Atualiza ByteTrack com pessoas
        people_np = np.array(people_dets, dtype=float) if people_dets else np.zeros((0,5), dtype=float)
        tracks = self.bt.update(people_np, (w,h))  # -> [(tid,x1,y1,x2,y2), ...]

        now = time.time()

        # Evento armas (cooldown global de armas)
        if weapons and (now - self.last_weapon_event) >= COOLDOWN:
            self.last_weapon_event = now
            print("[ALERTA] Arma detectada!")
            self._notify_clip(now, kind="weapon", frame=frame, extra={})

        # Cruzamentos de linhas por pés
        crossed_any = False
        for tid, x1,y1,x2,y2 in tracks:
            fx, fy = self._feet_point(x1,y1,x2,y2)
            if DRAW_FOOT_POINTS:
                cv2.circle(frame, (fx, fy), 5, (0,0,255), -1)
            # check cruzamento vs linhas A/B/C/L#
            prev = self.prev_foot.get(tid)
            if prev is not None:
                for ln in self.lines:
                    label = ln.get("name","")
                    if label not in ("A","B","C"):
                        continue
                    A = (ln["x1"], ln["y1"])
                    B = (ln["x2"], ln["y2"])
                    direction = crossed(prev, (fx,fy), A, B, CROSS_MARGIN)
                    if direction is not None:
                        print(f"[CROSS] ID{tid} cruzou {label}")
                        matched = self.seqmon.add(tid, label, now)
                        if matched is not None and (now - self.last_ambulant_event) >= COOLDOWN:
                            self.last_ambulant_event = now
                            pat_str = "→".join(matched)
                            print(f"[AMBULANTE] ID{tid} padrão {pat_str}!")
                            self._notify_clip(now, kind="ambulant", frame=frame,
                                              extra={"track_id": tid, "pattern": pat_str, "line": label})
                            crossed_any = True
            self.prev_foot[tid] = (fx, fy)

        # HUD (debug local opcional)
        for ln in self.lines:
            draw_line_with_name(frame, ln["x1"], ln["y1"], ln["x2"], ln["y2"], ln["name"])
        for _, x1,y1,x2,y2 in tracks:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)

        if self.cnt % (30 * max(1, FRAME_STRIDE)) == 0:
            fps = self.cnt / (time.time() - self.t0 + 1e-6)
            print(f"[YOLO] frames={self.cnt} fps~{fps:.1f} persons={len(tracks)} weapons={len(weapons)}")

        # (Opcional) visualize em uma janela X11 se existir DISPLAY
        if os.getenv("SHOW_WINDOW","0") in ("1","true","yes"):
            cv2.imshow("Deteccao", frame)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                os._exit(0)

        return Gst.FlowReturn.OK

# ------------------------- main -------------------------
def main():
    det = Detector()
    det.run()

if __name__ == "__main__":
    main()
