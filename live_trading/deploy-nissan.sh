#!/bin/bash
# Deploy CSP Strategy as a Cloud Run Job — Nissan account
#
# Usage:
#   ./deploy-nissan.sh              # Build + deploy
#   ./deploy-nissan.sh --build-only # Build image only (no deploy)
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project nissan-trading
#   Secrets created in Secret Manager:
#     ALPACA_API_KEY_V2, ALPACA_SECRET_KEY_V2

set -euo pipefail

PROJECT_ID="nissan-trading"
REGION="us-east1"
JOB_NAME="csp-strategy-nissan-paper"
IMAGE="us-east1-docker.pkg.dev/${PROJECT_ID}/csp-strategy/${JOB_NAME}"
GCS_BUCKET="nissan-options-data"

echo "=== CSP Strategy Deploy (Nissan) ==="
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Job:      ${JOB_NAME}"
echo "  Image:    ${IMAGE}"

# ── Ensure Artifact Registry repo exists ──
echo ""
echo "==> Ensuring Artifact Registry repository..."
gcloud artifacts repositories describe csp-strategy \
    --project="${PROJECT_ID}" \
    --location="${REGION}" > /dev/null 2>&1 || \
gcloud artifacts repositories create csp-strategy \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --repository-format=docker \
    --description="CSP strategy Docker images"

# ── Build and push Docker image ──
echo ""
echo "==> Building and pushing Docker image..."
gcloud builds submit \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --tag="${IMAGE}" \
    .

if [[ "${1:-}" == "--build-only" ]]; then
    echo ""
    echo "Build complete. Skipping deploy (--build-only)."
    exit 0
fi

# ── Create or update Cloud Run Job ──
echo ""
echo "==> Deploying Cloud Run Job..."
gcloud run jobs deploy "${JOB_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="${IMAGE}" \
    --task-timeout=28800 \
    --max-retries=0 \
    --memory=1Gi \
    --cpu=1 \
    --set-secrets="ALPACA_API_KEY=ALPACA_API_KEY_V2:latest,ALPACA_SECRET_KEY=ALPACA_SECRET_KEY_V2:latest" \
    --set-env-vars="\
PAPER_TRADING=true,\
STARTING_CASH=1000000,\
STORAGE_BACKEND=gcs,\
GCS_BUCKET_NAME=${GCS_BUCKET},\
GCS_PREFIX=paper-nissan,\
POLL_INTERVAL=60,\
MAX_CYCLES=0"

echo ""
echo "=== Deploy complete (Nissan) ==="
echo ""
echo "Run manually:    gcloud run jobs execute ${JOB_NAME} --region=${REGION}"
echo "View logs:       gcloud run jobs executions list --job=${JOB_NAME} --region=${REGION}"
echo ""
echo "To schedule (market hours, Mon-Fri 9:25 AM ET):"
echo "  gcloud scheduler jobs create http csp-nissan-daily-run \\"
echo "    --schedule='25 9 * * 1-5' \\"
echo "    --time-zone='America/New_York' \\"
echo "    --uri='https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run' \\"
echo "    --http-method=POST \\"
echo "    --oauth-service-account-email=\$(gcloud iam service-accounts list --format='value(email)' --filter='displayName:Compute Engine default' --project=${PROJECT_ID})"
