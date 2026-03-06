# src/utils/benchmark.py

import statistics
import numpy as np
from src.models.pipeline import PerceptionPipeline


def benchmark(pipeline: PerceptionPipeline, n_runs: int = 500):
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup runs — not counted
    print("Warming up...")
    for _ in range(20):
        pipeline.infer(dummy)

    # Timed runs
    print(f"Running {n_runs} iterations...")
    latencies = []
    for _ in range(n_runs):
        result = pipeline.infer(dummy)
        latencies.append(result['latency_ms'])

    latencies.sort()
    n = len(latencies)

    print("\n=== Latency Report ===")
    print(f"  Mean  : {statistics.mean(latencies):.2f}ms")
    print(f"  Stdev : {statistics.stdev(latencies):.2f}ms")
    print(f"  P50   : {latencies[n//2]:.2f}ms")
    print(f"  P95   : {latencies[int(0.95*n)]:.2f}ms")
    print(f"  P99   : {latencies[int(0.99*n)]:.2f}ms")
    print(f"  Max   : {latencies[-1]:.2f}ms")
    print(f"\nTarget <28ms P95: {'PASS' if latencies[int(0.95*n)] < 28 else 'FAIL'}")


if __name__ == '__main__':
    pipe = PerceptionPipeline(
        yolo_weights='weights/best.pt',
        sam_checkpoint='weights/sam_vit_h.pth',
    )
    benchmark(pipe)