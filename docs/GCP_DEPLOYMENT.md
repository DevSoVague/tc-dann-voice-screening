# TC-DANN: GCP Deployment Guide
> Complete step-by-step guide to train TC-DANN on GCP using 5 parallel H100 pods.

---

## Prerequisites

- GCP account with billing enabled
- `gcloud` CLI installed on your Mac → https://cloud.google.com/sdk/docs/install
- Docker installed on your Mac → https://docs.docker.com/get-docker/
- Your `my_model/` directory with `features/` and `phenotype/` subfolders
- Code files: `run.py`, `model.py`, `train.py`, `dataset.py`, `data_prep.py`, `layers.py`, `calibration.py`, `confidence.py`, `evaluate.py`, `requirements.txt`

---

## Variables Used Throughout This Guide

Replace these everywhere you see them:

| Placeholder | Your value |
|---|---|
| `YOUR_PROJECT_ID` | your GCP project id |
| `YOUR_EMAIL` | the Google account that owns the GCP project |
| `LOCAL_DATA_PATH` | local Bridge2AI-Voice root (same as `B2AI_DATA_ROOT`) |

---

## Part 0: Verify Data Locally First

**Do this before touching GCP. Fix any errors here.**

```bash
cd legacy_gcp
pip install -r requirements.txt
python run.py diagnose
```

Expected output: column names for each parquet, row counts per disease TSV, positive counts per disease. If this fails, fix it before proceeding.

---

## Part 1: Local Mac, Auth and Project Setup

```bash
# Login with the account that owns the GCP project
gcloud auth login

# Set the correct account and project
gcloud config set account YOUR_EMAIL
gcloud config set project YOUR_PROJECT_ID

# Verify
gcloud config list
```

---

## Part 2: Local Mac, Upload Data to GCS

Your dataset is 13GB and won't fit in Cloud Shell (5GB limit). Upload directly from your Mac to GCS.

```bash
# Create bucket
gcloud storage buckets create gs://tc-dann-data-YOUR_PROJECT_ID \
  --location=us-central1

# Upload features (~largest files, upload first)
cd $LOCAL_DATA_PATH

gcloud storage cp -r features/ \
  gs://tc-dann-data-YOUR_PROJECT_ID/my_model/features/

gcloud storage cp -r phenotype/ \
  gs://tc-dann-data-YOUR_PROJECT_ID/my_model/phenotype/
```

Upload time estimate at common speeds:
- 50 Mbps home wifi → ~35 min
- 100 Mbps CMU wifi → ~18 min
- 500 Mbps CMU wired → ~4 min

Verify upload completed:
```bash
gcloud storage ls gs://tc-dann-data-YOUR_PROJECT_ID/my_model/
# Should show: features/  phenotype/
```

---

## Part 3: Cloud Shell, Enable APIs

Open Cloud Shell at https://console.cloud.google.com and run:

```bash
gcloud config set project YOUR_PROJECT_ID

gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  storage-component.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com
```

---

## Part 4: Cloud Shell, Create GKE Cluster

```bash
# Base cluster (control plane only, cheap e2 node)
gcloud container clusters create tc-dann-cluster \
  --region us-central1 \
  --num-nodes 1 \
  --machine-type e2-standard-2 \
  --disk-type pd-standard \
  --disk-size 50GB \
  --no-enable-autoupgrade
```

Takes ~5 min. Then add the H100 GPU node pool:

```bash
gcloud container node-pools create h100-pool \
  --cluster tc-dann-cluster \
  --region us-central1 \
  --machine-type a3-highgpu-1g \
  --accelerator type=nvidia-h100-80gb,count=1 \
  --spot \
  --num-nodes 5 \
  --disk-type pd-standard \
  --disk-size 100GB \
  --no-enable-autoupgrade
```

Connect kubectl:

```bash
gcloud container clusters get-credentials tc-dann-cluster --region us-central1

# Verify, should show 6 nodes (1 control + 5 H100)
kubectl get nodes
```

---

## Part 5: Cloud Shell, Upload Code and Build Docker Image

**Upload your code files to Cloud Shell** using the upload button (three-dot menu → Upload). Upload all files from `my_model/files/`.

Then create the Dockerfile:

```bash
cd ~   # or wherever your files landed

cat > Dockerfile << 'EOF'
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git wget curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "run.py"]
EOF
```

Create Artifact Registry repo and build image:

```bash
PROJECT_ID=$(gcloud config get-value project)

# Create repo
gcloud artifacts repositories create tc-dann-repo \
  --repository-format=docker \
  --location=us-central1

# Auth Docker
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build image (takes ~5 min in Cloud Shell)
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/tc-dann-repo/model:latest .

# Push
docker push us-central1-docker.pkg.dev/$PROJECT_ID/tc-dann-repo/model:latest
```

---

## Part 6: Cloud Shell, IAM and Service Account

Pods need permission to read/write GCS:

```bash
PROJECT_ID=$(gcloud config get-value project)
BUCKET=tc-dann-data-$PROJECT_ID

# Create service account
gcloud iam service-accounts create tc-dann-sa \
  --display-name="TC-DANN SA"

# Grant bucket read + write
gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
  --member="serviceAccount:tc-dann-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Workload identity binding
gcloud iam service-accounts add-iam-policy-binding \
  tc-dann-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:$PROJECT_ID.svc.id.goog[default/tc-dann-ksa]"

# Kubernetes service account
kubectl create serviceaccount tc-dann-ksa

kubectl annotate serviceaccount tc-dann-ksa \
  iam.gke.io/gcp-service-account=tc-dann-sa@$PROJECT_ID.iam.gserviceaccount.com
```

---

## Part 7: Cloud Shell, Create PVC for Checkpoints

```bash
cat > pvc.yaml << 'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: checkpoints-pvc
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard-rwx
EOF

kubectl apply -f pvc.yaml
```

---

## Part 8: Cloud Shell, Launch 5 Parallel Training Jobs

```bash
PROJECT_ID=$(gcloud config get-value project)

cat > train_jobs.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: tc-dann-train
spec:
  completions: 5
  parallelism: 5
  completionMode: Indexed
  template:
    spec:
      serviceAccountName: tc-dann-ksa
      restartPolicy: OnFailure
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      initContainers:
        - name: download-data
          image: google/cloud-sdk:slim
          command:
            - bash
            - -c
            - gcloud storage cp -r gs://tc-dann-data-$PROJECT_ID/my_model/ /data/
          volumeMounts:
            - name: data-volume
              mountPath: /data
      containers:
        - name: trainer
          image: us-central1-docker.pkg.dev/$PROJECT_ID/tc-dann-repo/model:latest
          args:
            - "train"
            - "--root_dir=/data/my_model"
            - "--out_dir=/checkpoints/member_\$(JOB_COMPLETION_INDEX)"
            - "--epochs=40"
            - "--ensemble_size=1"
            - "--batch_size=32"
            - "--num_workers=4"
            - "--seed=\$(JOB_COMPLETION_INDEX)"
          env:
            - name: JOB_COMPLETION_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
            - name: PYTHONUNBUFFERED
              value: "1"
          resources:
            limits:
              nvidia.com/gpu: "1"
              memory: "32Gi"
            requests:
              nvidia.com/gpu: "1"
              memory: "24Gi"
          volumeMounts:
            - name: checkpoints
              mountPath: /checkpoints
            - name: data-volume
              mountPath: /data
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: checkpoints
          persistentVolumeClaim:
            claimName: checkpoints-pvc
        - name: data-volume
          emptyDir:
            sizeLimit: 100Gi
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
EOF

kubectl apply -f train_jobs.yaml
```

---

## Part 9: Cloud Shell, Monitor

```bash
# Watch pods start up (Ctrl+C to stop watching)
kubectl get pods -w

# Logs for pod 0
kubectl logs -f $(kubectl get pods | grep tc-dann-train | awk 'NR==1{print $1}')

# Tail last 5 lines from all pods
for pod in $(kubectl get pods | grep tc-dann-train | awk '{print $1}'); do
  echo "=== $pod ===" && kubectl logs $pod --tail=5
done

# Check overall job status
kubectl get jobs
```

Expected log output per pod:
```
[epoch 00] loss_d=0.693 grl_lambda=0.00 macro_auroc=0.512 worst_sub=0.498
[epoch 01] loss_d=0.641 grl_lambda=0.12 macro_auroc=0.671 worst_sub=0.603
...
```

Total expected time on H100 spot: **~40 min for all 5 members in parallel.**

---

## Part 10: Cloud Shell, Collect Checkpoints

Once all 5 pods show `Completed`:

```bash
PROJECT_ID=$(gcloud config get-value project)

# Copy from PVC to GCS
kubectl run copy-ckpts --image=google/cloud-sdk:slim --restart=Never \
  --overrides="{
    \"spec\": {
      \"volumes\": [{\"name\": \"ckpts\", \"persistentVolumeClaim\": {\"claimName\": \"checkpoints-pvc\"}}],
      \"containers\": [{
        \"name\": \"copy-ckpts\",
        \"image\": \"google/cloud-sdk:slim\",
        \"command\": [\"bash\", \"-c\", \"gcloud storage cp -r /checkpoints/ gs://tc-dann-data-$PROJECT_ID/checkpoints/\"],
        \"volumeMounts\": [{\"name\": \"ckpts\", \"mountPath\": \"/checkpoints\"}]
      }]
    }
  }"
```

---

## Part 11: Local Mac, Download and Evaluate

```bash
# Download checkpoints to your Mac
gcloud storage cp -r \
  gs://tc-dann-data-YOUR_PROJECT_ID/checkpoints/ \
  $LOCAL_DATA_PATH/checkpoints/

# Run evaluation locally
cd legacy_gcp
python run.py eval --ckpt_dir ../checkpoints/
```

Output lands in `eval_out/four_criteria.json`.

---

## Part 12: TEAR DOWN, DO THIS IMMEDIATELY AFTER

```bash
# Deletes cluster and stops ALL GPU billing
gcloud container clusters delete tc-dann-cluster \
  --region us-central1 --quiet
```

> **Set a phone reminder right now.** Forgetting this = unexpected bill.
> The GCS bucket keeps your data and checkpoints safe after cluster deletion.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `SSD_TOTAL_GB quota exceeded` | Add `--disk-type pd-standard` to cluster/pool create commands |
| `zsh: killed` locally | Add `--num_workers 0 --batch_size 4` |
| `403 does not have storage.objects.get` | Run `gcloud auth login` and switch to the correct account |
| `Already exists` on cluster create | Run `gcloud container clusters delete tc-dann-cluster --region us-central1 --quiet` first |
| Pod stuck in `Pending` | Run `kubectl describe pod POD_NAME`, usually a GPU quota issue |
| `enable_nested_tensor` warning | Safe to ignore, expected with Pre-LN transformer |
| `HF_TOKEN` warning | Safe to ignore, weights download fine without it |

---

## Cost Summary

| Resource | Rate | Est. Total |
|---|---|---|
| 5× H100 spot pods | ~$2.80/hr spot | ~$10 for full run |
| GCS storage (13GB) | ~$0.02/GB/month | ~$0.26/month |
| Cluster control plane | ~$0.10/hr | ~$1 for setup time |
| **Total** | | **~$12** |

---

## Order of Operations Cheatsheet

```
LOCAL MAC
  1. python run.py diagnose              ← verify data
  2. gcloud auth login                   ← correct account
  3. gcloud storage cp -r features/ ... ← upload data to GCS
  4. gcloud storage cp -r phenotype/ ...

CLOUD SHELL
  5. gcloud services enable ...          ← APIs
  6. cluster create + node pool create   ← GKE cluster
  7. upload code files via UI
  8. docker build + push                 ← Docker image
  9. IAM + service account setup
  10. kubectl apply -f pvc.yaml
  11. kubectl apply -f train_jobs.yaml   ← launch 5 pods
  12. kubectl get pods -w                ← watch logs

CLOUD SHELL (after training)
  13. copy checkpoints to GCS
  14. gcloud container clusters delete   ← STOP BILLING ← CRITICAL

LOCAL MAC
  15. gcloud storage cp checkpoints/     ← download
  16. python run.py eval                 ← evaluate
```
