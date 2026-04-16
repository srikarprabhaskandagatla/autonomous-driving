import argparse
import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Driving Perception System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train YOLOv9c on BDD100K")

    export_parser = subparsers.add_parser("export", help="Export YOLOv9c to ONNX (FP16)")
    export_parser.add_argument("--weights", default="weights/best.pt")

    serve_parser = subparsers.add_parser("serve", help="Start the inference API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark end-to-end pipeline latency")
    benchmark_parser.add_argument("--yolo-weights", default="weights/best.pt")
    benchmark_parser.add_argument("--sam-checkpoint", default="weights/sam_vit_h.pth")
    benchmark_parser.add_argument("--n-runs", type=int, default=500)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        from train import train
        train()

    elif args.command == "export":
        from src.models.export import export_to_onnx, validate_and_benchmark
        onnx_path = export_to_onnx(args.weights)
        validate_and_benchmark(onnx_path)

    elif args.command == "serve":
        uvicorn.run(
            "src.serving.server:app",
            host=args.host,
            port=args.port,
            workers=1,
        )

    elif args.command == "benchmark":
        from src.models.pipeline import PerceptionPipeline
        from src.utils.benchmark import run_benchmark, print_benchmark_report

        pipeline = PerceptionPipeline(
            yolo_weights=args.yolo_weights,
            sam_checkpoint=args.sam_checkpoint,
        )
        stats = run_benchmark(pipeline, n_runs=args.n_runs)
        print_benchmark_report(stats)


if __name__ == '__main__':
    main()
