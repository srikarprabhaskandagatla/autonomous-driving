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
        dynamic=False,
        simplify=True,
        opset=17,
        half=True,
        device=0,
    )

    onnx_path = str(Path(weights_path).with_suffix('.onnx'))
    return onnx_path


def build_ort_session(onnx_path: str) -> ort.InferenceSession:
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 3

    session = ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=[
            (
                'CUDAExecutionProvider',
                {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 ** 3,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
            ),
            'CPUExecutionProvider',
        ]
    )

    return session


def benchmark_ort_session(session: ort.InferenceSession, n_runs: int = 200) -> dict:
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float16)
    input_name = session.get_inputs()[0].name

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    n = len(latencies)

    return {
        'mean_ms': sum(latencies) / n,
        'p50_ms': latencies[n // 2],
        'p95_ms': latencies[int(0.95 * n)],
        'p99_ms': latencies[int(0.99 * n)],
    }


def validate_and_benchmark(onnx_path: str) -> ort.InferenceSession:
    session = build_ort_session(onnx_path)
    stats = benchmark_ort_session(session)

    print(f"ONNX Runtime Latency:")
    print(f"  Mean : {stats['mean_ms']:.2f}ms")
    print(f"  P50  : {stats['p50_ms']:.2f}ms")
    print(f"  P95  : {stats['p95_ms']:.2f}ms")
    print(f"  P99  : {stats['p99_ms']:.2f}ms")

    return session
