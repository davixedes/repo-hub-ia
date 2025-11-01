#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, glob, subprocess, tempfile, json, hashlib, sys, traceback, uuid
import numpy as np
import cv2
import boto3
from ultralytics import YOLO
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# ================== CONFIG (env sobrescreve) ==================
S3_BUCKET      = os.getenv("S3_BUCKET", "590183969767-detected-images")
S3_PREFIX      = os.getenv("S3_PREFIX", "weapons/")
AWS_REGION     = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
SQS_QUEUE_URL  = os.getenv("SQS_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/590183969767/detected-images")
SQS_IS_FIFO    = SQS_QUEUE_URL.endswith(".fifo")
SQS_GROUP      = os.getenv("SQS_GROUP", "camera-1")
CAMERA_ID      = os.getenv("CAMERA_ID", "train-camera-01")

ROLL_DIR       = os.getenv("ROLL_DIR", "/var/tmp/roll")   # onde ficam seg-0000000001.ts ...
SAVE_DIR       = os.getenv("SAVE_DIR", "/home/ubuntu/weapon_events")

PRE            = float(os.getenv("CLIP_PRE",  "5"))  # segundos antes
POST           = float(os.getenv("CLIP_POST", "5"))  # segundos depois
CONF           = float(os.getenv("CONF", "0.35"))
IOU            = float(os.getenv("IOU",  "0.45"))
DEVICE         = os.getenv("DEVICE", "cuda")         # "cuda" ou "cpu"
FRAME_STRIDE   = int(os.getenv("FRAME_STRIDE", "2")) # processa 1 a cada N frames

RTP_PORT       = int(os.getenv("RTP_PORT", "5004"))
PAYLOAD_PT     = int(os.getenv("PAYLOAD_PT", "98"))
CODEC_NAME     = os.getenv("CODEC_NAME", "H265")     # "H265" ou "H264"
LATENCY_MS     = int(os.getenv("LATENCY_MS", "120"))
COOLDOWN       = float(os.getenv("EVENT_COOLDOWN", "8.0"))  # seg entre eventos

# ================== AWS ==================
s3  = boto3.client("s3",  region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)

# ================== UTILS ==================
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

def pick_segments_by_mtime(center_ts, pre_s=5.0, post_s=5.0):
    """Seleciona .ts cujo mtime cobre [center_ts-pre, center_ts+post]"""
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
    """Concatena .ts → .mp4 sem re-encode (copy)."""
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

# ================== GStreamer ==================
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
    # splitmuxsink cria mpegtsmux internamente (definimos muxer=)
    return (
        f'udpsrc port={RTP_PORT} caps=application/x-rtp,media=video,encoding-name={CODEC_NAME},payload={PAYLOAD_PT},clock-rate=90000 ! '
        f'rtpjitterbuffer latency={LATENCY_MS} drop-on-latency=true do-lost=true ! '
        f'{depay_parse}'
        'tee name=t '
        # Ramo IA
        f't. ! queue leaky=2 max-size-buffers=30 ! {decode_branch} ! videoconvert ! video/x-raw,format=BGR ! '
        'appsink name=aisink emit-signals=true max-buffers=2 drop=true sync=false '
        # Ramo gravação (rolling .ts de 1s)
        f't. ! queue ! {parse_to_roll} ! '
        f'splitmuxsink name=rollsink muxer=mpegtsmux location="{ROLL_DIR}/seg-%010d.ts" max-size-time=1000000000 max-files=300 '
    )

PIPELINE = build_pipeline()
pipeline  = Gst.parse_launch(PIPELINE)
aisink    = pipeline.get_by_name("aisink")
if not aisink:
    print("ERRO: não achou appsink 'aisink'. Verifique GStreamer.")
    sys.exit(1)

# ================== YOLO ==================
# Se tiver pesos custom de armas, troque aqui.
model = YOLO("yolov8n.pt")
try:
    import torch
    if DEVICE == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass

# ================== LOOP IA ==================
cnt, t0, last_event = 0, time.time(), 0.0

def on_new_sample(sink):
    global cnt, t0, last_event
    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.OK

    buf  = sample.get_buffer()
    caps = sample.get_caps().get_structure(0)
    w, h = caps.get_value('width'), caps.get_value('height')
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.OK
    try:
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
    finally:
        buf.unmap(mapinfo)

    cnt += 1
    if FRAME_STRIDE > 1 and (cnt % FRAME_STRIDE) != 0:
        return Gst.FlowReturn.OK

    # YOLO: classes COCO 43=knife, 44=scissors
    res = model.predict(
        frame, imgsz=640, conf=CONF, iou=IOU, classes=[43,44],
        device=DEVICE, verbose=False
    )[0]

    found = 0
    if res.boxes is not None:
        found = len(res.boxes)

    if cnt % (30 * max(1, FRAME_STRIDE)) == 0:
        fps = cnt / (time.time() - t0 + 1e-6)
        print(f"[YOLO] frames={cnt} fps~{fps:.1f} objetos={found}")

    # evento
    if found > 0 and (time.time() - last_event) >= COOLDOWN:
        now = time.time()
        last_event = now
        print("[ALERTA] Objeto suspeito detectado!")
        # 1) Snapshot
        snaps_dir = os.path.join(SAVE_DIR, "snapshots")
        os.makedirs(snaps_dir, exist_ok=True)
        snap_name = f"weapon-{int(now)}.jpg"
        snap_path = os.path.join(snaps_dir, snap_name)
        try:
            cv2.imwrite(snap_path, frame)
        except Exception as e:
            log_err("[SNAP] falha", e)

        # 2) Clip ±5s
        time.sleep(POST + 0.7)  # garante que segmentos "depois" fecharam
        segs = pick_segments_by_mtime(now, PRE, POST)
        clips_dir = os.path.join(SAVE_DIR, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        clip_name = f"weapon-{int(now)}.mp4"
        out_mp4   = os.path.join(clips_dir, clip_name)
        ok = concat_ts_to_mp4(segs, out_mp4)

        # 3) Uploads + SQS
        # (envia pelo menos o snapshot; clipe se existir)
        media_sent = []
        # snapshot
        key_snap = f"{S3_PREFIX}snapshots/{snap_name}"
        meta_snap = upload_and_confirm(snap_path, key_snap)
        if meta_snap: media_sent.append(("snapshot", key_snap, meta_snap))

        # clip
        meta_clip = None
        key_clip  = None
        if ok:
            key_clip = f"{S3_PREFIX}clips/{clip_name}"
            meta_clip = upload_and_confirm(out_mp4, key_clip)
            if meta_clip: media_sent.append(("clip", key_clip, meta_clip))

        # payload mínimo + extras
        train_id = str(uuid.uuid4())
        base_payload = {
            "event": "weapon_detected",
            "train_id": train_id,
            "camera_id": CAMERA_ID,
            "bucket": S3_BUCKET,
            "ts_epoch": int(now),
            "window": {"pre_s": PRE, "post_s": POST},
            "codec": CODEC_NAME,
        }

        # envia 1 msg por mídia (snapshot/clip)
        for kind, s3_key, meta in media_sent:
            payload = dict(base_payload)
            payload.update({
                "kind": kind,          # snapshot | clip
                "s3_key": s3_key,      # <-- CAMPO OBRIGATÓRIO
                "metadata": meta
            })
            send_sqs(payload)

    return Gst.FlowReturn.OK

aisink.connect("new-sample", on_new_sample)

def main():
    try:
        ensure_dirs()
    except Exception as e:
        log_err("[DIR] erro ao preparar diretórios", e); sys.exit(1)

    pipeline.set_state(Gst.State.PLAYING)
    print(f"[GST] pipeline PLAYING (RTP {CODEC_NAME} porta {RTP_PORT}, PT={PAYLOAD_PT})")
    print("[GST] rtpjitterbuffer: drop-on-latency=TRUE do-lost=TRUE | appsink: sync=FALSE drop=TRUE")

    bus = pipeline.get_bus()
    try:
        while True:
            msg = bus.timed_pop_filtered(
                100 * Gst.MSECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS
            )
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
        pipeline.set_state(Gst.State.NULL)
        print("[GST] stopped")

if __name__ == "__main__":
    main()
