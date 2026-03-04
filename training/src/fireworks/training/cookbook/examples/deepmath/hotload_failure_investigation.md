# Hotload Failure Investigation - DeepMath GRPO Training

## Summary

Two consecutive training runs on `qwen3-4b` with training shape `qwen3-4b-minimum-h200` failed during hotload at steps 4-6. Both times the inference deployment pod became unhealthy and could not complete the hotload within the 600s timeout.

- **Run 1**: Pod preempted, rescheduled after ~7 min, then P2P hotload failed (stale trainer IP)
- **Run 2**: Pod crash-looping with `ENOSPC (No space left on device)` - disk full on the inference node

---

## Run 1: Pod Preemption + P2P Hotload Failure

### Key IDs

| Resource | ID |
|---|---|
| **Trainer Job ID** | `kb8vyphflnh3knf3` |
| **Deployment ID** | `deepmath-qwen3-4b-1772587041` |
| **Account** | `pyroworks` |
| **Base Model** | `accounts/fireworks/models/qwen3-4b` |
| **Training Shape** | `qwen3-4b-minimum-h200` |
| **Region** | `US_VIRGINIA_1` |
| **GCP Project** | `fw-ai-cp-prod` |
| **GCS Checkpoint Path** | `gs://fireworks-artifacts-pyroworks-417e69/rl-checkpoints/pyroworks/deepmath-qwen3-4b-1772587041/` |
| **WandB** | https://wandb.ai/myh97/grpo-tinker |

### Result
- Training ran 5 steps successfully (59.4% accuracy at step 5)
- Failed at step 6 during hotload

### Timeline

#### Step 6 checkpoint save (trainer side - successful)
- `01:52:33` - Trainer saves weights to `/tmp/rlor_checkpoints/step-6-c981166a`
- `01:52:47` - Delta checkpoint saved, upload to GCS started
- `01:52:49` - GCS upload: `gcloud storage cp -r ... gs://fireworks-artifacts-pyroworks-417e69/rl-checkpoints/pyroworks/deepmath-qwen3-4b-1772587041/step-6-c981166a/`
- `01:52:53` - GCS upload complete (8 files, 0.49 GB)

#### Inference pod eviction (~01:53)
- `01:53:23` - K8s scheduling failure: `0/472 nodes are available`
  - Causes: Insufficient GPU (322 nodes), node affinity mismatch (134), untolerated taints (training-worker, bad-node, eval, disk-pressure), insufficient memory (2 nodes)
- `01:53:29` through `02:00:25` - Pod remains unschedulable for ~7 minutes

#### Pod restart and P2P hotload attempt
- `02:02:56` - New inference pod starts: `pyroworks-deepmath-qwen3-4b-1772587041-7977465994-qp7wr`
- `02:02:56.412` - Hotload lifecycle: transitioning `None -> step-0-base-c981166a`
- `02:02:56.484` - **"No longer accepting requests to start p2p hot loading step-6-c981166a"**
- `02:02:56.495` - **"p2p hot load is enabled. Calling with len(ledger)=7 and target_snapshot_identity='step-6-c981166a'"**
- `02:02:56.616` - Worker processes invoke `p2p_hot_load` via NIXL/UCX
- P2P fails because the trainer peer IP (from before preemption) is stale/unreachable

### GCP Log Queries

```bash
# Trainer logs
gcloud logging read '"kb8vyphflnh3knf3" AND ("hotload" OR "snapshot" OR "save_state" OR "step-6")' --limit=50 --format=json --freshness=24h --project=fw-ai-cp-prod

# Inference/deployment logs
gcloud logging read '"deepmath-qwen3-4b-1772587041" AND ("hot_load" OR "p2p" OR "nixl" OR "preempt" OR "evict" OR "unschedulable")' --limit=100 --format=json --freshness=24h --project=fw-ai-cp-prod

# K8s scheduling events
gcloud logging read '"deepmath-qwen3-4b-1772587041" AND ("nodes are available" OR "Preemption" OR "unschedulable")' --limit=50 --format=json --freshness=24h --project=fw-ai-cp-prod
```

---

## Run 2: Disk Full (ENOSPC) Crash Loop

### Key IDs

| Resource | ID |
|---|---|
| **Policy Trainer Job ID** | `rqjiqp5rcaoaayd8` |
| **Reference Trainer Job ID** | `oubbpaqx8hc8vmgc` |
| **Deployment ID** | `deepmath-qwen3-4b-1772608390` |
| **Account** | `pyroworks` |
| **Base Model** | `accounts/fireworks/models/qwen3-4b` |
| **Training Shape** | `qwen3-4b-minimum-h200` |
| **Region** | `US_VIRGINIA_1` |
| **GCP Project** | `fw-ai-cp-prod` |
| **WandB** | https://wandb.ai/myh97/grpo-tinker/runs/7dy87b3s |

### Result
- Training ran 3 steps successfully (84.4% accuracy at step 3)
- Failed at step 4 during hotload

### Timeline

#### Hotload attempt at step 4
- `07:40:25` - First HTTP 500 from hotload gateway, followed by sustained HTTP 503s
- `07:40:26` through `07:45:25` - 14 retries over 300s, all HTTP 503
- `07:45:25` - SDK logs: `Error checking hotload status: 503 Service Unavailable`
- `07:45:30` through `07:50:30` - Another round of 14 retries, all HTTP 503
- `07:50:35` - **Hotload timeout (600s)**: `ERROR: Hotload did not complete within 600s`

#### Inference pod crash-loop (disk full)
The serving container was crash-looping with `ENOSPC` during the entire hotload window:
- `07:42:32` - `OSError: [Errno 28] No space left on device` (restart 1)
- `07:43:36` - `OSError: [Errno 28] No space left on device` (restart 2)
- `07:45:00` - `OSError: [Errno 28] No space left on device` (restart 3)
- `07:47:04` - `OSError: [Errno 28] No space left on device` (restart 4)
- `07:50:32` - `OSError: [Errno 28] No space left on device` (restart 5)

Each restart cycle: server starts (~2s) -> immediately hits disk full -> crashes -> backoff -> restart.

### GCP Log Queries

```bash
# Serving container ENOSPC errors
gcloud logging read 'resource.labels.pod_name:"deepmath-qwen3-4b-1772608390" AND resource.labels.container_name="serving" AND ("No space" OR "Errno 28" OR "ENOSPC")' --limit=30 --format=json --freshness=24h --project=fw-ai-cp-prod

# All serving container errors
gcloud logging read 'resource.labels.pod_name:"deepmath-qwen3-4b-1772608390" AND resource.labels.container_name="serving" AND severity>=ERROR' --limit=50 --format=json --freshness=24h --project=fw-ai-cp-prod
```

---

## Open Questions

### 1. Why is P2P hotload enabled instead of fuse-based Alluxio loading? (Run 1)

The deployment uses `hot_load_bucket_type="FW_HOSTED"` (default). We expected this to use the local Alluxio fuse mount to read checkpoints from GCS. However, the inference server logs explicitly show `p2p hot load is enabled` and uses `reload_parameters_and_buffers_p2p` via NIXL/UCX to transfer weights directly from the trainer.

**Question:** What determines whether P2P vs fuse-based hotload is used? Is this controlled by the training shape, deployment shape, or a server-side flag? The Python SDK has no P2P-related configuration options.

### 2. P2P hotload is fragile to pod rescheduling (Run 1)

When the inference pod gets preempted and rescheduled to a new node, P2P hotload breaks because:
- The new pod tries to connect to the old trainer IP (stale after pod migration)
- There's no fallback to fuse-based loading when P2P fails

**Question:** Should there be a fallback mechanism from P2P to fuse-based loading when the peer is unreachable?

### 3. Why is the inference pod running out of disk space? (Run 2)

The inference pod for `qwen3-4b` (a ~4B parameter model, ~8GB weights) is hitting `ENOSPC` on startup. Possible causes:
- Alluxio cache filling up the local disk with checkpoint deltas from multiple hotload steps
- Shared node disk pressure from other deployments
- Ephemeral storage limit too low for the deployment shape

**Question:** What is the ephemeral storage allocation for the `qwen3-4b-minimum-h200` deployment shape? Is checkpoint cleanup happening between hotload steps?
