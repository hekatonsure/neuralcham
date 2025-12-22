#!/bin/bash
# Sync local code + data to runpod
# Usage: ./sync_to_pod.sh <pod_id>

POD_ID=${1:?"Usage: $0 <pod_id>"}

echo "=== Syncing to pod $POD_ID ==="

# Create archive of project (excluding large files)
tar -czf /tmp/neuralcham.tar.gz \
    --exclude='checkpoints/*' \
    --exclude='outputs/*' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    .

# Send to pod
runpodctl send /tmp/neuralcham.tar.gz

echo ""
echo "On the pod, run:"
echo "  runpodctl receive"
echo "  tar -xzf neuralcham.tar.gz"
echo "  ./scripts/pod_setup.sh"
echo "  ./scripts/run_all.sh"
