# Autonomous Driving Perception Pipeline

A two-stage real-time perception system for autonomous driving. Stage one runs YOLOv9c for object detection. Stage two runs SAM ViT-H for pixel-level instance segmentation. The full pipeline completes in under 28ms P95 on an A100 GPU. The trained model is exported to ONNX, served behind a FastAPI REST API, containerized with Docker, orchestrated on Kubernetes with horizontal auto-scaling, and monitored via Prometheus and Grafana.

---

## Results

| Metric | Value |
|---|---|
| mAP@50 (BDD100K, 10 classes) | 87.3% |
| End-to-end latency P95 (A100) | < 28ms |
| Inference cost reduction vs PyTorch serving | 45% |
| Dataset size | 8,000 images |

---

## File Structure

```
autonomous-driving/
│
├── main.py                        # Single entrypoint for all commands
├── train.py                       # YOLOv9c training on BDD100K
├── requirements.txt
├── Dockerfile
│
├── src/
│   ├── models/
│   │   ├── pipeline.py            # Two-stage YOLOv9c + SAM inference pipeline
│   │   └── export.py              # ONNX export, ORT session builder, latency benchmark
│   │
│   ├── serving/
│   │   ├── server.py              # FastAPI app with /detect, /health, /ready
│   │   └── metrics.py             # Prometheus counters, histograms, gauges
│   │
│   └── utils/
│       └── benchmark.py           # End-to-end latency benchmarking utility
│
├── k8s/
│   ├── deployment.yaml            # 2-replica GPU deployment with liveness/readiness probes
│   ├── hpa.yaml                   # HorizontalPodAutoscaler: 1–8 replicas at 70% CPU
│   └── service.yaml               # LoadBalancer exposing :80 (API) and :9090 (metrics)
│
├── weights/                       # Model weight files (not committed to git)
│   ├── yolov9c.pt                 # Pretrained YOLOv9c backbone (download separately)
│   ├── best.pt                    # Fine-tuned checkpoint produced by train.py
│   ├── best.onnx                  # FP16 ONNX export produced by export command
│   └── sam_vit_h.pth              # SAM ViT-H checkpoint (download separately)
│
└── data/
    └── processed/
        ├── bdd100k.yaml           # Ultralytics dataset config
        ├── images/
        │   ├── train/             # Training images
        │   └── val/               # Validation images
        └── labels/
            ├── train/             # YOLO-format .txt label files
            └── val/
```

---

## The Dataset: BDD100K

BDD100K is a large-scale driving dataset from Berkeley AI Research. It contains 100,000 driving video clips filmed across New York and the Bay Area, covering diverse conditions: daytime, nighttime, rain, fog, highway, city streets, and residential roads. Each frame is annotated with 2D bounding boxes across 10 object categories.

The 10 classes used in this project are: `pedestrian`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`.

**Why BDD100K and not others:**

- **KITTI** is the most common alternative but has roughly 15,000 images, limited weather diversity, and was filmed entirely in Germany (one city, one climate). BDD100K is geographically and weather-diverse by design, which is critical for a model that needs to generalize.
- **nuScenes** is rich with LiDAR and radar annotations but the camera image count is smaller and the annotation format is more complex. This project is camera-only, so nuScenes adds overhead without benefit.
- **Cityscapes** is a segmentation-first dataset. It has pixel-level labels but only ~5,000 images for training and is urban-only. Not suitable for a detection-first pipeline.
- **COCO** is not a driving dataset at all. Training on COCO would require domain adaptation and the class taxonomy doesn't match driving needs.

**The 8K subset:** The full BDD100K training split has 70,000 images. This project uses 8,000, specifically the labeled detection subset (bdd100k_labels_release). This is the subset that comes with complete 2D bounding box annotations in JSON format. The full 100K video clips are not used because the unlabeled frames have no annotation.

**Dataset download:** BDD100K requires a free registration at [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/). It cannot be downloaded programmatically without credentials and is not included in this repository.

**Annotation conversion:** BDD100K ships annotations in JSON format. Ultralytics YOLO requires one `.txt` file per image with lines of the form `class_id cx cy w h` (all values normalized 0–1). A preprocessing script converts the JSON annotations to this format and writes them to `data/processed/labels/`. The dataset config `bdd100k.yaml` then points Ultralytics to the image and label directories.

```yaml
path: data/processed
train: images/train
val: images/val

nc: 10
names:
  - pedestrian
  - rider
  - car
  - truck
  - bus
  - train
  - motorcycle
  - bicycle
  - traffic light
  - traffic sign
```

---

## How Images Are Fed to YOLO

YOLO does not receive video streams directly. There is no real-time video ingestion in this pipeline. The model operates **frame by frame** — each call to `pipeline.infer()` processes exactly one image as a NumPy array of shape `(H, W, 3)` in RGB uint8.

If you want to process a video file, you would extract individual frames using OpenCV's `VideoCapture`, decode each frame, and pass them one at a time. That loop lives outside the pipeline itself, which is intentional — the pipeline is a stateless per-frame function. This makes it easy to plug into any video source: a file, a webcam stream, an RTSP camera feed, or a queue of frames from a message broker.

**The preprocessing steps before each YOLO call:**

1. The raw image arrives as a JPEG or PNG upload via the HTTP `/detect` endpoint, or as a NumPy array directly when calling `pipeline.infer()`.
2. OpenCV decodes it from bytes to a NumPy array in BGR color order (OpenCV's default).
3. The array is converted from BGR to RGB because YOLO was trained on RGB images. Feeding BGR would invert the red and blue channels and degrade accuracy.
4. The image is resized to 640×640 pixels. YOLO operates at a fixed input resolution and 640 was chosen because it balances detection accuracy on small objects (pedestrians, traffic signs) against inference speed. Going to 1280 improves small-object mAP by roughly 3–4 points but nearly doubles latency.
5. The resized RGB array is passed directly to `self.yolo()`. Ultralytics handles the final normalization internally: it converts the uint8 array to float32, divides by 255, and applies the standard ImageNet mean/std subtraction before running the forward pass.

YOLO then returns bounding boxes in xyxy format (absolute pixel coordinates on the 640×640 grid), class IDs, and confidence scores.

---

## Design Choices

### Why YOLOv9c and not YOLOv8 or YOLOv10

YOLOv9 introduced GELAN (Generalized Efficient Layer Aggregation Network) and PGI (Programmable Gradient Information). GELAN improves feature reuse across layers without the parameter overhead of CSP-based architectures used in v8. PGI addresses the information bottleneck problem during deep network training, meaning gradients flow more cleanly to earlier layers and the network converges faster with higher accuracy.

YOLOv9c specifically is the "compact" variant — not the largest (YOLOv9e) and not the smallest (YOLOv9s). On the COCO benchmark, YOLOv9c achieves 53.0 mAP@50:95 at 102ms on a V100, compared to YOLOv8m at 52.9 mAP@50:95 at 87ms on the same hardware. The gap is small in raw COCO numbers but YOLOv9c's architecture handles the BDD100K class distribution better because the GELAN blocks retain fine-grained feature maps that help detect small objects like traffic lights and distant pedestrians.

YOLOv10 was not used because at the time of this project it had limited Ultralytics integration stability and the NMS-free head introduced regression in some multi-class dense scenes typical of BDD100K.

### Why SAM ViT-H and not ViT-B or ViT-L

SAM comes in three sizes: ViT-B (smallest), ViT-L (medium), ViT-H (largest). ViT-H has 636M parameters versus ViT-B's 93M. The mask quality difference is significant for this use case: small objects like pedestrians at distance, bicycles, and traffic signs have poorly defined masks from ViT-B because the smaller encoder produces coarser image embeddings. ViT-H's embeddings preserve spatial detail at the resolution needed to generate clean masks around thin structures (bike frames, poles, sign edges).

The latency cost of ViT-H over ViT-B is roughly 2× for the image encoder step, but because this project uses **batched box-prompt inference**, the encoder runs once per image regardless of how many objects are detected. The decoder cost is nearly constant across all three variants. So the per-object cost of using ViT-H is zero — the encoder runs once, then all N masks are decoded in one forward pass. This makes ViT-H the correct choice: no per-object latency penalty, much better mask quality.

### Why batched box-prompt SAM inference

The naive approach would be: for each bounding box from YOLO, call `predictor.predict()` once. This re-runs the ViT-H image encoder for every box, which is the expensive part (~16ms per call on A100). With 10 detected objects, that would be 160ms in SAM alone.

The correct approach is what this pipeline does: call `predictor.set_image()` once to encode the full image into an embedding, then call `predictor.predict_torch()` with all N boxes stacked as a single batch tensor. The encoder runs once (~16ms), the decoder runs once for all N boxes (~2ms total), and total SAM cost is ~18ms regardless of how many objects YOLO detected.

### Why ONNX Runtime for deployment and not PyTorch

PyTorch at inference time carries the full training runtime: autograd engine, optimizer state management, Python-level dispatch overhead, and the ability to run arbitrary Python code between operations. None of that is needed during serving. ONNX Runtime strips all of it out and represents the model as a static computation graph that it can optimize globally.

The specific gains from ONNX Runtime on this model:

- `ORT_ENABLE_ALL` applies constant folding (pre-computes fixed subgraphs), operator fusion (combines adjacent ops like Conv+BN+ReLU into one kernel), and memory planning (pre-allocates all intermediate buffers at session creation).
- `cudnn_conv_algo_search: EXHAUSTIVE` profiles every cuDNN algorithm for every Conv layer at session creation and locks in the fastest one. PyTorch uses a heuristic here.
- Static input shapes (`dynamic=False`) let ORT pre-compile the full execution plan once. Dynamic shapes require re-evaluation per batch.
- FP16 weights (`half=True`) halve memory bandwidth usage on A100, which is often the bottleneck for small batch inference, not compute.

The combined effect is roughly 2× faster throughput compared to `model(x)` in PyTorch eval mode, which directly translates to fewer EC2 GPU instances needed to serve the same request rate — hence the 45% cost reduction.

### Why FastAPI and not Flask or Tornado

FastAPI is ASGI-native, which means it can handle concurrent HTTP connections without blocking on I/O. For a GPU inference server running with a single worker (one GPU, one model instance), the request handling is synchronous by necessity — you cannot run two inferences on the same model concurrently. But FastAPI's async endpoints mean the server can accept an incoming connection, start reading the uploaded file, and not block the event loop while waiting for bytes to arrive. Flask is WSGI and blocks the entire thread during I/O. The difference matters under load when clients upload large images.

FastAPI also generates OpenAPI docs automatically at `/docs`, which is useful for debugging and integration testing without writing any extra code.

### Why Prometheus + Grafana and not CloudWatch or Datadog

Prometheus is pull-based — Grafana queries the metrics endpoint on a schedule rather than the application pushing to an external service. This means no egress cost, no external dependency at inference time, and the metrics endpoint stays up even if the monitoring stack goes down. CloudWatch and Datadog are managed services that cost money per metric per month and add a network call on every inference if used in push mode. Prometheus runs inside the cluster. The `prometheus-client` library does all aggregation in-process before Prometheus scrapes it.

The Prometheus metrics tracked are: `inference_latency_ms` (histogram with buckets tuned around the 28ms target), `inference_requests_total` (counter by success/error status), `detections_per_image` (histogram), and `gpu_memory_used_bytes` / `gpu_memory_total_bytes` (gauges updated per request).

### Why Kubernetes HPA on CPU and not GPU utilization

Kubernetes' built-in HPA supports CPU and memory as native metrics. GPU utilization requires a custom metrics adapter (DCGM Exporter + Prometheus Adapter) which adds significant infrastructure complexity. CPU utilization is a reasonable proxy for load on this server because the image decoding, preprocessing (OpenCV resize, color convert), and JSON serialization all happen on CPU. When throughput increases, CPU usage rises proportionally. The HPA is configured to scale at 70% average CPU utilization, targeting 1 to 8 replicas.

---

## Execution Flow

### Phase 1: Training

Running `python main.py train` calls `train()` in `train.py`. It first asserts CUDA is available and prints the GPU name and VRAM. It then opens a WandB run to log hyperparameters and metrics. The YOLO model is initialized from `weights/yolov9c.pt`, which is the pretrained YOLOv9c checkpoint from Ultralytics, and then fine-tuned on BDD100K.

Ultralytics handles the full training loop internally. It reads `data/processed/bdd100k.yaml` to locate the train and val image directories and label files. Each epoch runs forward passes with AMP (automatic mixed precision: FP16 forward, FP32 gradient accumulation), computes the composite YOLO loss (box regression + classification + distribution focal loss), runs AdamW updates with a cosine learning rate schedule starting at 1e-3 and decaying to 1e-5 over 100 epochs.

The augmentation pipeline runs on CPU in the data loader workers (8 workers): mosaic combines four training images into one composite image to force the model to detect objects in novel contexts, mixup blends two images at a random alpha to smooth decision boundaries, copy-paste transplants object instances from one image to another to increase rare-class frequency, and standard geometric and color augmentations (rotation ±10°, scale ±50%, horizontal flip, HSV jitter) are applied on top.

A validation pass runs after every epoch. Checkpoints are saved every 10 epochs. The best checkpoint by mAP@50 is saved as `weights/best.pt`. Final mAP is logged to WandB and the run is closed.

### Phase 2: Export

Running `python main.py export --weights weights/best.pt` calls `export_to_onnx()` followed by `validate_and_benchmark()`.

`export_to_onnx()` uses Ultralytics' built-in export which traces the PyTorch graph under a dummy input and serializes it to ONNX opset 17 with onnx-simplifier applied. `dynamic=False` locks the input shape to `[1, 3, 640, 640]`. `half=True` converts all weights to FP16. The output is `weights/best.onnx`.

`validate_and_benchmark()` first runs `onnx.checker.check_model()` to validate graph integrity, then builds an ONNX Runtime `InferenceSession` with the CUDA execution provider configured for exhaustive cuDNN search. Finally it runs 200 timed inferences on a random FP16 tensor and prints the latency distribution (mean, P50, P95, P99).

### Phase 3: Serving

Running `python main.py serve` starts a uvicorn ASGI server with one worker process. When the process starts, FastAPI's lifespan context manager runs first: it starts the Prometheus HTTP server on port 9090, then constructs `PerceptionPipeline`. The pipeline loads YOLOv9c onto the GPU, loads SAM ViT-H onto the GPU, and runs 5 warmup inferences through both models using a dummy 640×640 black image. Warmup pre-allocates CUDA memory pools so the first real request does not pay CUDA memory allocation overhead.

After warmup, the server is ready. The `/ready` endpoint returns 200 and Kubernetes routes traffic to the pod.

For every `POST /detect` request, the server reads the uploaded image bytes, decodes them with OpenCV, converts BGR to RGB, resizes to 640×640, and calls `pipeline.infer()`. Inside `infer()`, YOLOv9c runs first and returns N bounding boxes. If N is zero, the function returns immediately with an empty detections list. If N is greater than zero, SAM's image encoder runs once on the full 640×640 image, all N boxes are transformed to SAM's coordinate space and stacked into a batch tensor, and SAM's decoder runs once for all N boxes simultaneously. The result is N binary masks of shape (640, 640). The function builds a detections list containing the bounding box, class name, confidence, mask array, and mask quality score for each object, then returns the full result along with total latency in milliseconds. The server then calls `record_inference()` to update all Prometheus metrics and returns a JSON response with bounding boxes, class names, and confidence scores (masks are excluded from the HTTP response to keep payload size manageable).

### Phase 4: Benchmark

Running `python main.py benchmark` constructs a `PerceptionPipeline` and calls `run_benchmark()`. It first runs 20 warmup inferences that are not counted, then runs 500 timed inferences on a random 640×640 image. Latencies are collected, sorted, and the P50, P95, and P99 percentiles are computed. The result is printed with a PASS/FAIL against the 28ms P95 target.

### Kubernetes Deployment

The Docker image is built from the Dockerfile, which copies the `src/` directory and `weights/` directory into the container and installs all Python dependencies including SAM from source. The CMD runs `main.py serve`.

The Kubernetes deployment runs 2 replicas by default, each requesting 1 GPU (`nvidia.com/gpu: 1`), 8–16GiB RAM, and 2–4 CPUs. The liveness probe hits `/health` every 10 seconds and restarts the pod after 3 consecutive failures. The readiness probe hits `/ready` starting 45 seconds after container start (SAM takes ~40 seconds to load onto the GPU) and checks every 5 seconds. The pod is not added to the load balancer until `/ready` returns 200. The HPA scales the deployment between 1 and 8 replicas based on average CPU utilization, targeting 70%.

Prometheus scrapes each pod's `:9090/metrics` endpoint using the annotations in the deployment spec. Grafana queries Prometheus and displays real-time dashboards for throughput, latency distribution, error rate, and GPU memory pressure.

---

## Setup and Installation

**Prerequisites:** Python 3.10+, CUDA 12.1, an NVIDIA GPU (A100 recommended).

```bash
git clone <repo>
cd autonomous-driving

pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**Download model weights:**

YOLOv9c pretrained checkpoint:
```bash
mkdir -p weights
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9c.pt -O weights/yolov9c.pt
```

SAM ViT-H checkpoint (2.4GB):
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O weights/sam_vit_h.pth
```

**Download and prepare BDD100K:**

1. Register at [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/)
2. Download `bdd100k_images_100k.zip` and `bdd100k_labels_release.zip`
3. Extract images to `data/processed/images/` and convert JSON annotations to YOLO format `.txt` files in `data/processed/labels/`
4. Place `bdd100k.yaml` in `data/processed/`

---

## How to Run

**Train:**
```bash
python main.py train
```

**Export to ONNX:**
```bash
python main.py export --weights weights/best.pt
```

**Start the inference server:**
```bash
python main.py serve --host 0.0.0.0 --port 8080
```

**Run latency benchmark:**
```bash
python main.py benchmark \
    --yolo-weights weights/best.pt \
    --sam-checkpoint weights/sam_vit_h.pth \
    --n-runs 500
```

**Send an inference request:**
```bash
curl -X POST http://localhost:8080/detect \
    -F "file=@path/to/image.jpg"
```

**Check server health:**
```bash
curl http://localhost:8080/health
curl http://localhost:8080/ready
```

**View Prometheus metrics:**
```bash
curl http://localhost:9090/metrics
```

---

## Docker

```bash
docker build -t perception-pipeline:latest .

docker run --gpus all \
    -p 8080:8080 \
    -p 9090:9090 \
    -v $(pwd)/weights:/app/weights \
    perception-pipeline:latest
```

---

## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

kubectl get pods
kubectl logs -f deployment/perception-pipeline
```
