import statistics
import numpy as np
from src.models.pipeline import PerceptionPipeline


def run_benchmark(pipeline: PerceptionPipeline, n_runs: int = 500) -> dict:
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    for _ in range(20):
        pipeline.infer(dummy)

    latencies = []
    for _ in range(n_runs):
        result = pipeline.infer(dummy)
        latencies.append(result['latency_ms'])

    latencies.sort()
    n = len(latencies)

    return {
        'mean_ms': statistics.mean(latencies),
        'stdev_ms': statistics.stdev(latencies),
        'p50_ms': latencies[n // 2],
        'p95_ms': latencies[int(0.95 * n)],
        'p99_ms': latencies[int(0.99 * n)],
        'max_ms': latencies[-1],
        'p95_pass': latencies[int(0.95 * n)] < 28,
    }


def print_benchmark_report(stats: dict):
    print("\n=== Latency Report ===")
    print(f"  Mean  : {stats['mean_ms']:.2f}ms")
    print(f"  Stdev : {stats['stdev_ms']:.2f}ms")
    print(f"  P50   : {stats['p50_ms']:.2f}ms")
    print(f"  P95   : {stats['p95_ms']:.2f}ms")
    print(f"  P99   : {stats['p99_ms']:.2f}ms")
    print(f"  Max   : {stats['max_ms']:.2f}ms")
    print(f"\nTarget <28ms P95: {'PASS' if stats['p95_pass'] else 'FAIL'}")
