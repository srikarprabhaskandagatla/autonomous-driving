# src/models/export.py

import time
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO


def export_to_onnx(weights_path: str) -> str:
    model = YOLO(weights_path)

    model.export(
        format='onnx',
        imgsz=640,
        dynamic=False,    # Static input shape = enables more ORT graph optimizations
        simplify=True,    # Runs onnx-simplifier: removes redundant nodes
        opset=17,         # Opset 17 = latest stable, required for some GELAN ops
        half=True,        # FP16 weights
        device=0,
    )

    onnx_path = weights_path.replace('.pt', '.onnx')
    print(f"Exported: {onnx_path}")
    return onnx_path


def validate_onnx(onnx_path: str) -> ort.InferenceSession:
    # Validate model integrity
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model integrity check passed.")

    # Build optimized ORT session
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level       = 3  # Suppress verbose ORT logs

    session = ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=[
            (
                'CUDAExecutionProvider',
                {
                    'device_id':              0,
                    'arena_extend_strategy':  'kNextPowerOfTwo',
                    'gpu_mem_limit':          4 * 1024 ** 3,  # 4GB cap
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',   # Best algo for static shapes
                    'do_copy_in_default_stream': True,
                }
            ),
            'CPUExecutionProvider',  # Fallback only
        ]
    )

    # Benchmark
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float16)
    input_name = session.get_inputs()[0].name

    latencies = []
    for _ in range(200):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    print(f"ONNX Runtime Latency:")
    print(f"  Mean : {sum(latencies)/len(latencies):.2f}ms")
    print(f"  P50  : {latencies[100]:.2f}ms")
    print(f"  P95  : {latencies[190]:.2f}ms")
    print(f"  P99  : {latencies[198]:.2f}ms")

    return session


if __name__ == '__main__':
    onnx_path = export_to_onnx('weights/best.pt')
    validate_onnx(onnx_path)