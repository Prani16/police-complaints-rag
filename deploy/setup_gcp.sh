#!/usr/bin/env bash
# One-time GCP setup for the police-complaints RAG project.
#   PROJECT_ID=my-project ./deploy/setup_gcp.sh
# Requires: gcloud CLI logged in as a project Owner/Editor.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-asia-south1}"
REPO="${REPO:-rag}"

gcloud config set project "$PROJECT_ID"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com iam.googleapis.com

echo "==> Artifact Registry repo '$REPO' in $REGION"
gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" \
    --description="police-complaints RAG images"

echo "==> Service accounts"
for sa in rag-backend-sa rag-frontend-sa; do
  gcloud iam service-accounts describe "$sa@$PROJECT_ID.iam.gserviceaccount.com" >/dev/null 2>&1 || \
    gcloud iam service-accounts create "$sa" --display-name="$sa"
done

echo "==> Gemini API key -> Secret Manager (secret: gemini-api-key)"
if ! gcloud secrets describe gemini-api-key >/dev/null 2>&1; then
  read -r -s -p "Paste your Gemini API key (from aistudio.google.com/apikey): " KEY; echo
  printf '%s' "$KEY" | gcloud secrets create gemini-api-key --data-file=- --replication-policy=automatic
fi
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:rag-backend-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor >/dev/null

echo "==> Admin dashboard password -> Secret Manager (secret: admin-password)"
if ! gcloud secrets describe admin-password >/dev/null 2>&1; then
  read -r -s -p "Choose a password for the /admin evaluation dashboard: " APW; echo
  printf '%s' "$APW" | gcloud secrets create admin-password --data-file=- --replication-policy=automatic
fi
for sa in rag-backend-sa rag-frontend-sa; do
  gcloud secrets add-iam-policy-binding admin-password \
    --member="serviceAccount:$sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor >/dev/null
done

echo "==> Permissions for Cloud Build (deploys to Cloud Run)"
# Newer projects run Cloud Build as the Compute default SA, older ones as the
# legacy Cloud Build SA - grant both so either works.
for builder in "$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
               "$PROJECT_NUMBER@cloudbuild.gserviceaccount.com"; do
  for role in roles/run.admin roles/iam.serviceAccountUser roles/artifactregistry.writer roles/logging.logWriter; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$builder" \
      --role="$role" --condition=None >/dev/null 2>&1 || true
  done
done

echo "Done. Now run: PROJECT_ID=$PROJECT_ID ./deploy/deploy.sh"
