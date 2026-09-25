import os
import argparse
from .utils import resolve_fuse_path
from .data import localize_and_compile_data

from hierarchical_yolo.train import train_curriculum

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Dataset Hierarchical YOLO Trainer on Vertex AI")
    
    # Vertex Submission Arguments
    parser.add_argument(
        "--datasets",
        nargs='+',
        required=True,
        help="List of space-separated GCS URIs pointing to staging datasets."
    )
    
    # Passed through to train_curriculum
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="yolov8n.pt")
    parser.add_argument("--shallow_epochs", type=int, default=2)
    parser.add_argument("--final_epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    
    args = parser.parse_args()

    # 1. Clean dataset URIs
    dataset_uris = [uri.strip() for uri in args.datasets if uri.strip()]
    if not dataset_uris:
        raise ValueError("No valid dataset URIs provided in --datasets.")

    # 2. Vertex AI Output Resolution
    # Vertex AI sets AIP_MODEL_DIR based on your job config. 
    # We use this as our root output directory so Ultralytics writes directly back to GCS.
    aip_model_dir = os.getenv("AIP_MODEL_DIR")
    if not aip_model_dir:
        raise EnvironmentError("AIP_MODEL_DIR not set. Are you running inside Vertex AI Custom Training?")
        
    gcs_output_dir = resolve_fuse_path(aip_model_dir)

    # 3. Localize & Orchestrate Data
    workspace_dir = localize_and_compile_data(dataset_uris)
    
    # 4. Train
    train_curriculum(
        workspace_dir=workspace_dir,
        model_dir=gcs_output_dir,
        project_name=args.project_name,
        base_model=args.base_model,
        shallow_epochs=args.shallow_epochs,
        final_epochs=args.final_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        val=True,
        resume=False # Set to True if you want to support resuming interrupted Vertex runs later
    )
