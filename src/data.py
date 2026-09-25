import os
import shutil
from typing import List

from .utils import resolve_fuse_path
from hierarchical_yolo.dataset_merger import build_mega_dataset
from hierarchical_yolo.data_orchestrator import build_hierarchical_workspace

def localize_and_compile_data(dataset_uris: List[str]) -> str:
    """
    Downloads datasets from GCS to local ephemeral storage, merges them,
    and compiles the final hierarchical YOLO workspace using hard-links.
    
    Returns the path to the compiled YOLO workspace.
    """
    print("\n" + "="*60)
    print("☁️ Phase 1: Localizing Data from Cloud Storage")
    print("="*60)
    
    raw_staging_dir = "/tmp/raw_datasets"
    mega_dataset_dir = "/tmp/mega_dataset"
    yolo_workspace_dir = "/tmp/yolo_workspace"
    
    os.makedirs(raw_staging_dir, exist_ok=True)
    local_dataset_paths = []
    
    # 1. Fetch from GCS FUSE to local SSD
    for i, uri in enumerate(dataset_uris):
        fuse_path = resolve_fuse_path(uri.strip())
        local_dest = os.path.join(raw_staging_dir, f"dataset_{i}")
        
        print(f"\nCopying Dataset {i+1}/{len(dataset_uris)}")
        print(f"Source (FUSE): {fuse_path}")
        print(f"Destination (Local): {local_dest}")
        
        if not os.path.exists(fuse_path):
            raise FileNotFoundError(f"GCS FUSE path does not exist: {fuse_path}")
            
        shutil.copytree(fuse_path, local_dest)
        local_dataset_paths.append(local_dest)
        
    # 2. Merge into Mega-Dataset (using your existing library)
    print("\n" + "="*60)
    print("🧬 Phase 2: Building Unified Mega-Dataset")
    print("="*60)
    build_mega_dataset(source_dirs=local_dataset_paths, target_dir=mega_dataset_dir)
    
    # 3. Compile Hierarchical Workspace (using your existing library)
    print("\n" + "="*60)
    print("🏗️ Phase 3: Compiling Hierarchical YOLO Workspace")
    print("="*60)
    build_hierarchical_workspace(source_dir=mega_dataset_dir, workspace_dir=yolo_workspace_dir)
    
    return yolo_workspace_dir
