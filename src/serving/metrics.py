# src/serving/metrics.py

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import torch


INFERENCE_LATENCY = Histogram(
    'inference_latency_ms',
    'End-to-end inference latency in milliseconds',
    buckets=[5, 10, 15, 20, 25, 28, 30, 35, 40, 50, 75, 100]
)

REQUESTS_TOTAL = Counter(
    'inference_requests_total',
    'Total inference requests received',
    ['status']
)

DETECTIONS_PER_IMAGE = Histogram(
    'detections_per_image',
    'Number of objects detected per image',
    buckets=[0, 1, 2, 5, 10, 20, 50, 100]
)

GPU_MEMORY_USED_BYTES = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory currently allocated in bytes'
)

GPU_MEMORY_TOTAL_BYTES = Gauge(
    'gpu_memory_total_bytes',
    'Total GPU memory in bytes'
)


def record_inference(latency_ms: float, n_detections: int, success: bool):
    INFERENCE_LATENCY.observe(latency_ms)
    REQUESTS_TOTAL.labels(status='success' if success else 'error').inc()
    DETECTIONS_PER_IMAGE.observe(n_detections)

    if torch.cuda.is_available():
        GPU_MEMORY_USED_BYTES.set(torch.cuda.memory_allocated(0))
        GPU_MEMORY_TOTAL_BYTES.set(
            torch.cuda.get_device_properties(0).total_memory
        )


def start_metrics_server(port: int = 9090):
    start_http_server(port)
    print(f"Prometheus metrics available at :{port}/metrics")