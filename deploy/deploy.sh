#!/usr/bin/env bash
# Build + deploy both services with Cloud Build.
#   PROJECT_ID=my-project ./deploy/deploy.sh
#   GEMINI_MODEL=gemini-4-argon MIN_INSTANCES=1 PROJECT_ID=my-project ./deploy/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-asia-south1}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.5-flash-lite}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"

gcloud builds submit . --project="$PROJECT_ID" --config=deploy/cloudbuild.yaml \
  --substitutions="_REGION=$REGION,_GEMINI_MODEL=$GEMINI_MODEL,_MIN_INSTANCES=$MIN_INSTANCES"

URL=$(gcloud run services describe rag-frontend --project="$PROJECT_ID" --region="$REGION" --format='value(status.url)')
echo
echo "App:  $URL"
echo "CLI:  (backend is private) use the web app, or run the CLI against a local backend"
