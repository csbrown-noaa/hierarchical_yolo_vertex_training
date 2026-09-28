#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Initialize variables
PROJECT_ID=""
REGION=""
IMAGE_URI=""
STAGING_BUCKET=""
DISK_SIZE="100" # Default 100GB
DRY_RUN=0

# Hardware configuration (Defaults)
MACHINE_TYPE="n1-standard-8"
ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
ACCELERATOR_COUNT="1"

# Function to display help menu
usage() {
  echo "Usage: $0 [GCP OPTIONS] -- [CONTAINER ARGUMENTS]"
  echo ""
  echo "Submits a Multi-Dataset Hierarchical Training Job to Vertex AI."
  echo ""
  echo "GCP Options (Required):"
  echo "  -p, --project          Google Cloud Project ID"
  echo "  -r, --region           GCP Region (e.g., us-central1)"
  echo "  -i, --image-uri        Full Artifact Registry URI of the training container"
  echo "  -b, --bucket           GCS Staging Bucket for outputs (e.g., gs://my-bucket/training_runs)"
  echo ""
  echo "GCP Hardware Options (Optional):"
  echo "  -s,  --disk-size       Boot disk size in GB (default: 100)"
  echo "  -mt, --machine-type    Compute instance type (default: n1-standard-8)"
  echo "  -at, --accelerator     Accelerator/GPU type (default: NVIDIA_TESLA_T4)"
  echo "  -ac, --accel-count     Number of accelerators (default: 1)"
  echo "  --dry                  Print the gcloud command without executing it"
  echo "  -h,  --help            Display this help message and exit"
  echo ""
  echo "----------------------------------------------------------------------"
  echo "Container Arguments (MUST follow '--'):"
  echo "----------------------------------------------------------------------"
  echo "  --project_name         Name of the run/project (Required)"
  echo "  --datasets             Space-separated GCS dataset URIs (Required, no quotes needed!)"
  echo "  --base_model           Base model architecture (Optional, default: yolov8n.pt)"
  echo ""
  echo "Example:"
  echo "  $0 -p my-proj -r us-central1 -i my-image -b gs://bucket -- \\"
  echo "     --project_name my_run --datasets gs://data1 gs://data2 --base_model yolo.pt"
  echo ""
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--project) PROJECT_ID="$2"; shift 2 ;;
    -r|--region) REGION="$2"; shift 2 ;;
    -i|--image-uri) IMAGE_URI="$2"; shift 2 ;;
    -b|--bucket) STAGING_BUCKET="$2"; shift 2 ;;
    -s|--disk-size) DISK_SIZE="$2"; shift 2 ;;
    -mt|--machine-type) MACHINE_TYPE="$2"; shift 2 ;;
    -at|--accelerator) ACCELERATOR_TYPE="$2"; shift 2 ;;
    -ac|--accel-count) ACCELERATOR_COUNT="$2"; shift 2 ;;
    --dry) DRY_RUN=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    --) 
      shift
      PASSTHROUGH_ARGS=("$@")
      break
      ;;
    *) echo "Error: Unknown option: $1"; usage; exit 1 ;;
  esac
done

# Ensure all required variables are set
MISSING_ARGS=0
if [[ -z "$PROJECT_ID" ]]; then echo "Error: --project is required."; MISSING_ARGS=1; fi
if [[ -z "$REGION" ]]; then echo "Error: --region is required."; MISSING_ARGS=1; fi
if [[ -z "$IMAGE_URI" ]]; then echo "Error: --image-uri is required."; MISSING_ARGS=1; fi
if [[ -z "$STAGING_BUCKET" ]]; then echo "Error: --bucket is required."; MISSING_ARGS=1; fi

if [[ $MISSING_ARGS -eq 1 ]]; then
  echo ""
  usage
  exit 1
fi

if [ ${#PASSTHROUGH_ARGS[@]} -eq 0 ]; then
  echo "Error: No container arguments found. You must include '--' followed by python args."
  usage
  exit 1
fi

JOB_NAME="yolo-train-$(date +%Y%m%d-%H%M%S)"

echo ""
echo "Submitting Custom Training Job: ${JOB_NAME}..."
echo "Project:      ${PROJECT_ID}"
echo "Region:       ${REGION}"
echo "Output:       ${STAGING_BUCKET}/${JOB_NAME}"
echo "Image:        ${IMAGE_URI}"
echo "Hardware:     ${MACHINE_TYPE} w/ ${ACCELERATOR_COUNT}x${ACCELERATOR_TYPE}"
echo "Disk Size:    ${DISK_SIZE}GB"
echo "Container Args: ${PASSTHROUGH_ARGS[*]}"
echo "--------------------------------------------------------"

# ------------------------------------------------------------------------------
# 1. Build the Hardware YAML Config
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

# ------------------------------------------------------------------------------
# 2. Convert Passthrough Array to Comma-Separated String for gcloud --args
# ------------------------------------------------------------------------------
SAVE_IFS="$IFS"
IFS=","
CONTAINER_ARGS="${PASSTHROUGH_ARGS[*]}"
IFS="$SAVE_IFS"

# ------------------------------------------------------------------------------
# 3. Submit the Job
# ------------------------------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[DRY RUN] The following command would be executed:"
  echo "--------------------------------------------------------"
  echo "gcloud ai custom-jobs create \\"
  echo "  --project=\"${PROJECT_ID}\" \\"
  echo "  --region=\"${REGION}\" \\"
  echo "  --display-name=\"${JOB_NAME}\" \\"
  echo "  --config=\"${TEMP_CONFIG}\" \\"
  echo "  --args=\"${CONTAINER_ARGS}\""
  echo "--------------------------------------------------------"
else
  gcloud ai custom-jobs create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --config="${TEMP_CONFIG}" \
    --args="${CONTAINER_ARGS}"
fi

# Clean up the temporary config file
rm "${TEMP_CONFIG}"

if [ "$DRY_RUN" -eq 0 ]; then
  echo ""
  echo "Job submitted successfully! Monitor logs in the GCP Console under Vertex AI -> Training."
  echo ""
fi
