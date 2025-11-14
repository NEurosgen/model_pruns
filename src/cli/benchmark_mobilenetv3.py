#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Benchmark quantized MobileNetV3-Large on CPU and Raspberry Pi.
# Features:
# - PyTorch (quantized) and TFLite backends
# - Video stream or synthetic images
# - End-to-end timing (optional) and model-only timing
# - Warmup, percentiles, FPS, CPU threading controls
#
# Usage examples
# - Torch + webcam (end-to-end timing): 
#     python3 benchmark_mobilenetv3.py --backend torch --video 0 --include-preproc
# - Torch + file video, 4 CPU threads:
#     python3 benchmark_mobilenetv3.py --backend torch --video path/to/video.mp4 --threads 4
# - TFLite + webcam:
#     python3 benchmark_mobilenetv3.py --backend tflite --video 0 --tflite-model model.tflite
# - Synthetic images, model-only timing (fast microbenchmark):
#     python3 benchmark_mobilenetv3.py --backend torch --iters 500
#
# Notes:
# - For PyTorch: provide a quantized model via --torch-model if you have a custom .pt/.pth. 
#   Otherwise the script tries to build a dynamically-quantized MobileNetV3-Large.
# - For TFLite: provide a .tflite via --tflite-model.

import argparse, os, time, sys, json
from pathlib import Path
import numpy as np

# Optional imports guarded
try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
    torch.backends.quantized.engine = "fbgemm"
    from torchvision import transforms
    import torchvision
except Exception:
    torch = None
    transforms = None
    torchvision = None

def log(s):
    print(s, flush=True)

def build_preprocess(size=224):
    # CV2 -> RGB, resize, to tensor, normalize (MobileNetV3 ImageNet)
    mean = [0.5025, 0.4846, 0.5003]
    std  = [0.1574, 0.1490, 0.1549]

    pipeline = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    def _pp(frame_bgr):
        import cv2 as _cv2
        img = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
        x = pipeline(img).unsqueeze(0).numpy()  # NCHW numpy
        return x
    return _pp

def attempt_build_quantized_mnv3_large_torch():
    if torchvision is None:
        raise RuntimeError("torchvision not available")
    try:
        model = torchvision.models.mobilenet_v3_large(weights=None)
        model.eval()
        if torch is not None:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to create MobileNetV3-Large: {e}")

def load_torch_model(path="/home/temp/MyDir/Projects/aspdfpwjfpwejfwpefwef/saved_model/mnv3_dynamic_int8_scripted.pt", map_location="cpu"):
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    if path is not None:
        qmodel = torch.jit.load(path)
        return qmodel
    else: 
        print("Path is None")

def to_torch_tensor(np_batch):
    x = torch.from_numpy(np_batch)
    return x

def torch_infer(model, x, sync=True):
    with torch.no_grad():
        y = model(x)
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    return y

def bench_torch(args):
    if args.video is not None and cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for video mode. Install opencv-python.")
    if torch is None:
        raise RuntimeError("PyTorch is required for --backend torch.")

    # CPU backend setup
    try:
        torch.set_num_interop_threads(max(1, args.threads))
        torch.set_num_threads(max(1, args.threads))
    except Exception:
        pass
    if hasattr(torch, "backends") and hasattr(torch.backends, "quantized") and hasattr(torch.backends.quantized, "engine"):
        try:
            torch.backends.quantized.engine = "fbgemm"
        except Exception:
            pass

    model = load_torch_model()
    model.eval()
    model.to("cpu")

    preprocess = build_preprocess()

    timings = []
    n_frames = 0
    start_total = time.perf_counter()

    # Warmup (synthetic)
    if args.warmup > 0:
        xw = np.random.randn(args.batch, 3,244, 244).astype(np.float32)
        xw_t = to_torch_tensor(xw)
        for _ in range(args.warmup):
            _ = torch_infer(model, xw_t, sync=False)

    if args.video is not None:
        cap = cv2.VideoCapture(0 if str(args.video) == "0" else str(args.video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {args.video}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.include_preproc:
                t0 = time.perf_counter()
                x = preprocess(frame)  # NCHW float32 numpy
                x_t = to_torch_tensor(x)
                _ = torch_infer(model, x_t, sync=False)
                t1 = time.perf_counter()
                timings.append(t1 - t0)  # end-to-end
            else:
                x = preprocess(frame)
                x_t = to_torch_tensor(x)
                t0 = time.perf_counter()
                _ = torch_infer(model, x_t, sync=False)
                t1 = time.perf_counter()
                timings.append(t1 - t0)  # model-only
            n_frames += 1
            if args.max_frames and n_frames >= args.max_frames:
                break
        cap.release()
    else:
        for _ in range(args.iters):
            x = np.random.randn(args.batch, 3,244, 244).astype(np.float32)
            x_t = to_torch_tensor(x)
            t0 = time.perf_counter()
            _ = torch_infer(model, x_t, sync=False)
            t1 = time.perf_counter()
            timings.append(t1 - t0)
            n_frames += args.batch

    total = time.perf_counter() - start_total
    report_and_save(timings, n_frames, total, args, backend="torch")

def load_tflite_interpreter(tflite_path, threads=1):
    try:
        from tflite_runtime.interpreter import Interpreter
        from tflite_runtime.interpreter import load_delegate
    except Exception:
        try:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
            load_delegate = None
        except Exception as e:
            raise RuntimeError("Neither tflite_runtime nor TensorFlow Lite is available.") from e

    interpreter = Interpreter(model_path=tflite_path, num_threads=max(1, threads))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def tflite_infer(interpreter, input_details, output_details, x_nchw):
    x = x_nchw  # NCHW float32
    x = np.transpose(x, (0, 2, 3, 1))  # -> NHWC
    in_dt = input_details[0]["dtype"]
    if in_dt == np.uint8:
        scale, zero = input_details[0].get("quantization", (1.0, 0))
        xq = x / scale + zero
        xq = np.clip(xq, 0, 255).astype(np.uint8)
        x = xq
    elif in_dt == np.int8:
        scale, zero = input_details[0].get("quantization", (1.0, 0))
        xq = x / scale + zero
        xq = np.clip(xq, -128, 127).astype(np.int8)
        x = xq
    else:
        x = x.astype(in_dt)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]["index"])
    return y

def bench_tflite(args):
    if args.video is not None and cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for video mode. Install opencv-python.")
    if not args.tflite_model:
        raise RuntimeError("--tflite-model is required for --backend tflite.")

    interpreter, input_details, output_details = load_tflite_interpreter(args.tflite_model, args.threads)
    preprocess = build_preprocess(244)

    timings = []
    n_frames = 0
    start_total = time.perf_counter()

    # Warmup
    if args.warmup > 0:
        xw = np.random.randn(args.batch, 3, 244, 244).astype(np.float32)
        for _ in range(args.warmup):
            _ = tflite_infer(interpreter, input_details, output_details, xw)

    if args.video is not None:
        cap = cv2.VideoCapture(0 if str(args.video) == "0" else str(args.video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {args.video}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.include_preproc:
                t0 = time.perf_counter()
                x = preprocess(frame)  # NCHW float32
                _ = tflite_infer(interpreter, input_details, output_details, x)
                t1 = time.perf_counter()
                timings.append(t1 - t0)  # end-to-end
            else:
                x = preprocess(frame)
                t0 = time.perf_counter()
                _ = tflite_infer(interpreter, input_details, output_details, x)
                t1 = time.perf_counter()
                timings.append(t1 - t0)  # model-only
            n_frames += 1
            if args.max_frames and n_frames >= args.max_frames:
                break
        cap.release()
    else:
        for _ in range(args.iters):
            x = np.random.randn(args.batch, 3,244, 244).astype(np.float32)
            t0 = time.perf_counter()
            _ = tflite_infer(interpreter, input_details, output_details, x)
            t1 = time.perf_counter()
            timings.append(t1 - t0)
            n_frames += args.batch

    total = time.perf_counter() - start_total
    report_and_save(timings, n_frames, total, args, backend="tflite")

def report_and_save(timings, n_frames, total, args, backend):
    if not timings:
        log("No timings collected.")
        return
    lat_ms = np.array(timings) * 1000.0
    fps = n_frames / sum(timings) if sum(timings) > 0 else 0.0
    report = {
        "backend": backend,
        "mode": "video" if args.video is not None else "synthetic",
        "include_preproc": bool(args.include_preproc),
        "threads": int(args.threads),
        "batch": int(args.batch),
        "input_size": int(244),
        "n_frames": int(n_frames),
        "wall_time_sec": float(total),
        "avg_latency_ms": float(lat_ms.mean()),
        "p50_ms": float(np.percentile(lat_ms, 50)),
        "p90_ms": float(np.percentile(lat_ms, 90)),
        "p95_ms": float(np.percentile(lat_ms, 95)),
        "p99_ms": float(np.percentile(lat_ms, 99)),
        "throughput_fps": float(fps),
    }
    print(json.dumps(report, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("backend,mode,include_preproc,threads,batch,input_size,n_frames,wall_time_sec,avg_ms,p50_ms,p90_ms,p95_ms,p99_ms,throughput_fps\n")
        f.write("{backend},{mode},{include_preproc},{threads},{batch},{input_size},{n_frames},{wall:.6f},{avg:.3f},{p50:.3f},{p90:.3f},{p95:.3f},{p99:.3f},{fps:.3f}\n".format(
            backend=report["backend"],
            mode=report["mode"],
            include_preproc=int(report["include_preproc"]),
            threads=report["threads"],
            batch=report["batch"],
            input_size=report["input_size"],
            n_frames=report["n_frames"],
            wall=report["wall_time_sec"],
            avg=report["avg_latency_ms"],
            p50=report["p50_ms"],
            p90=report["p90_ms"],
            p95=report["p95_ms"],
            p99=report["p99_ms"],
            fps=report["throughput_fps"],
        ))
    print(f"Saved metrics to: {out}")

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark MobileNetV3-Large (quantized) on CPU/Raspberry Pi")
    p.add_argument("--backend", choices=["torch", "tflite"], required=True, help="Inference backend")
    p.add_argument("--tflite-model", type=str, default=None, help="Path to a TFLite .tflite model (required for tflite backend)")
    p.add_argument("--video", type=str, default=None, help="Video path or '0' for webcam. If omitted, synthetic input is used.")
    p.add_argument("--iters", type=int, default=200, help="Iterations for synthetic mode")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")
    p.add_argument("--input-size", type=int, default=224, help="Model input resolution (square)")
    p.add_argument("--batch", type=int, default=1, help="Batch size (torch only effectively benefits; tflite runs 1)")
    p.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1), help="CPU threads to use")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p.add_argument("--include-preproc", action="store_true", help="Measure end-to-end (preproc + model). Otherwise model-only.")
    p.add_argument("--out", type=str, default="benchmark_metrics.csv", help="Where to save a one-line CSV with summary metrics")
    return p.parse_args()

def main():
    args = parse_args()
    if args.backend == "torch":
        bench_torch(args)
    else:
        bench_tflite(args)

if __name__ == "__main__":

    main()

#/home/temp/MyDir/Projects/ml_things/Results/results/processed_sample_EfficientNetB0_TL.avi