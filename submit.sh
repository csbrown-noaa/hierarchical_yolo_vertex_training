#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Initialize variables
PROJECT_ID=""
REGION=""
IMAGE_URI=""
DATASETS=""
STAGING_BUCKET=""
PROJECT_NAME="hierarchical_run"
MODEL_ARCH="yolov8n.pt"
DISK_SIZE="100" # Default 100GB

# Hardware configuration (Defaults)
MACHINE_TYPE="n1-standard-8" 
ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
ACCELERATOR_COUNT="1"

# Function to display help menu
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Submits a Multi-Dataset Hierarchical Training Job to Vertex AI."
  echo ""
  echo "Options (Required):"
  echo "  -p, --project          Google Cloud Project ID"
  echo "  -r, --region           GCP Region (e.g., us-central1)"
  echo "  -i, --image-uri        Full Artifact Registry URI of the training container"
  echo "  -d, --datasets         Comma-separated list of GCS dataset URIs (e.g., gs://bucket/d1,gs://bucket/d2)"
  echo "  -b, --bucket           GCS Staging Bucket for outputs (e.g., gs://my-bucket/training_runs)"
  echo ""
  echo "Options (Optional):"
  echo "  -n,  --name            Project name for output (default: hierarchical_run)"
  echo "  -m,  --model-arch      Base model architecture (default: yolov8n.pt)"
  echo "  -s,  --disk-size       Boot disk size in GB (default: 100)"
  echo "  -mt, --machine-type    Compute instance type (default: n1-standard-8)"
  echo "  -at, --accelerator     Accelerator/GPU type (default: NVIDIA_TESLA_T4)"
  echo "  -ac, --accel-count     Number of accelerators (default: 1)"
  echo "  -h,  --help            Display this help message and exit"
  echo ""
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--project) PROJECT_ID="$2"; shift 2 ;;
    -r|--region) REGION="$2"; shift 2 ;;
    -i|--image-uri) IMAGE_URI="$2"; shift 2 ;;
    -d|--datasets) DATASETS="$2"; shift 2 ;;
    -b|--bucket) STAGING_BUCKET="$2"; shift 2 ;;
    -n|--name) PROJECT_NAME="$2"; shift 2 ;;
    -m|--model-arch) MODEL_ARCH="$2"; shift 2 ;;
    -s|--disk-size) DISK_SIZE="$2"; shift 2 ;;
    -mt|--machine-type) MACHINE_TYPE="$2"; shift 2 ;;
    -at|--accelerator) ACCELERATOR_TYPE="$2"; shift 2 ;;
    -ac|--accel-count) ACCELERATOR_COUNT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: Unknown option: $1"; usage; exit 1 ;;
  esac
done

# Ensure all required variables are set
MISSING_ARGS=0
if [[ -z "$PROJECT_ID" ]]; then echo "Error: --project is required."; MISSING_ARGS=1; fi
if [[ -z "$REGION" ]]; then echo "Error: --region is required."; MISSING_ARGS=1; fi
if [[ -z "$IMAGE_URI" ]]; then echo "Error: --image-uri is required."; MISSING_ARGS=1; fi
if [[ -z "$DATASETS" ]]; then echo "Error: --datasets is required."; MISSING_ARGS=1; fi
if [[ -z "$STAGING_BUCKET" ]]; then echo "Error: --bucket is required."; MISSING_ARGS=1; fi

if [[ $MISSING_ARGS -eq 1 ]]; then
  echo ""
  usage
  exit 1
fi

JOB_NAME="yolo-train-$(date +%Y%m%d-%H%M%S)"

echo ""
echo "Submitting Custom Training Job: ${JOB_NAME}..."
echo "Project:      ${PROJECT_ID}"
echo "Region:       ${REGION}"
echo "Datasets:     ${DATASETS}"
echo "Output:       ${STAGING_BUCKET}/${JOB_NAME}"
echo "Model:        ${MODEL_ARCH}"
echo "Image:        ${IMAGE_URI}"
echo "Hardware:     ${MACHINE_TYPE} w/ ${ACCELERATOR_COUNT}x ${ACCELERATOR_TYPE}"
echo "Disk Size:    ${DISK_SIZE}GB"
echo "--------------------------------------------------------"

# ------------------------------------------------------------------------------
# Vertex AI CLI Workaround:
# The gcloud CLI doesn't support setting disk size natively via string flags.
# We dynamically generate a temporary YAML config to pass the disk requirements
# and the baseOutputDirectory to avoid CLI flag conflicts.
# ------------------------------------------------------------------------------
TEMP_CONFIG="tmp_vertex_config_${JOB_NAME}.yaml"

cat <<EOF > "${TEMP_CONFIG}"
baseOutputDirectory:
  outputUriPrefix: ${STAGING_BUCKET}/${JOB_NAME}
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
      acceleratorType: ${ACCELERATOR_TYPE}
      acceleratorCount: ${ACCELERATOR_COUNT}
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: ${DISK_SIZE}
    containerSpec:
      imageUri: ${IMAGE_URI}
EOF

# Submit the job to Vertex AI using the generated config
gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --config="${TEMP_CONFIG}" \
  --args="--datasets=${DATASETS}" \
  --args="--project_name=${PROJECT_NAME}" \
  --args="--base_model=${MODEL_ARCH}"

# Clean up the temporary config file
rm "${TEMP_CONFIG}"

echo ""
echo "Job submitted successfully! Monitor logs in the GCP Console under Vertex AI -> Training."
echo ""

