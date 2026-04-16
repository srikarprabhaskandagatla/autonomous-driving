import numpy as np
import cv2
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.models.pipeline import PerceptionPipeline
from src.serving.metrics import record_inference, start_metrics_server


pipeline: PerceptionPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    start_metrics_server(port=9090)
    pipeline = PerceptionPipeline(
        yolo_weights='weights/best.pt',
        sam_checkpoint='weights/sam_vit_h.pth',
        device='cuda',
    )
    yield
    del pipeline


app = FastAPI(
    title="Autonomous Driving Perception API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Must upload an image file")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))

        result = pipeline.infer(image)

        record_inference(
            latency_ms=result['latency_ms'],
            n_detections=len(result['detections']),
            success=True,
        )

        return JSONResponse({
            "latency_ms": round(result['latency_ms'], 2),
            "n_detections": len(result['detections']),
            "detections": [
                {
                    "bbox": d['bbox'],
                    "class_name": d['class_name'],
                    "confidence": round(d['confidence'], 4),
                }
                for d in result['detections']
            ]
        })

    except HTTPException:
        raise
    except Exception as e:
        record_inference(latency_ms=0, n_detections=0, success=False)
        raise HTTPException(status_code=500, detail=str(e))
