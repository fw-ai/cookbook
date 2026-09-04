# Secure training — BYOB, CMEK, and secure RFT setup

*Source of truth: live [Secure Training](https://docs.fireworks.ai/fine-tuning/secure-fine-tuning.md), [data handling](https://docs.fireworks.ai/guides/security_compliance/data_handling.md). Contact Fireworks for onboarding emails and OIDC audience values.*

Use when automating BYOB dataset registration, CMEK key registration, or secure RFT with external storage. Human-readable policy and retention tables stay on the docs page.

## BYOB (bring your own bucket)

Fireworks reads datasets in-place during training; no copy is persisted on Fireworks-managed storage.

```bash
firectl dataset create {DATASET_NAME} --external-url gs://bucket/path/data.jsonl
# or s3://... or azure://...

firectl sftj create \
  --dataset accounts/{ACCOUNT}/datasets/{DATASET_NAME} \
  --base-model accounts/fireworks/models/{MODEL} \
  --output-model {OUTPUT}
```

RFT: same `--external-url` on dataset; use `firectl rftj create` with `--aws-iam-role` when required.

### GCS IAM (summary)

Grant three principals (emails provided at BYOB onboarding):

1. Fireworks control plane SA — custom role with `storage.buckets.getIamPolicy`
2. Fireworks inference SA — `roles/storage.objectViewer`
3. Your company Fireworks account email (`firectl account get`) — `roles/storage.objectViewer`

Revoke bindings after the job completes.

### AWS S3 (summary)

- Trust policy: `accounts.google.com` federated principal, `sts:AssumeRoleWithWebIdentity`, audience = Fireworks-provided OIDC value
- Policy: `s3:GetObject`, `s3:ListBucket` on dataset prefix
- Pass `--aws-iam-role arn:aws:iam::...:role/...` on job create
- Alternative: credentials secret (rotate regularly; federation preferred)

### Azure Blob (summary)

- Prefer Workload Identity Federation
- Or `--azure-credentials-secret accounts/.../secrets/...`

## CMEK (AWS KMS today)

Envelope encryption: Fireworks wraps per-resource DEKs with your KMS key. Revoke/disable key → data unreadable at rest.

### Fireworks federation identity (AWS)

| Field | Value |
|---|---|
| OIDC issuer | `https://accounts.google.com` |
| Subject | `108606366655288854355` |
| Audience | Your Fireworks account ID |

Grant only `kms:Encrypt` and `kms:Decrypt` on the key via IAM role + web-identity trust.

### Register key (draft CLI — verify `--help`)

```bash
firectl account external-keys create \
  --cloud aws \
  --kms-key "<KMS_KEY_ARN>"
firectl account external-keys list
```

After registration, `firectl dataset create` and `firectl sftj create` encrypt client-side transparently.

### Rotation / revocation

- New KMS key version → new wraps automatically; old data still decrypts
- Disable or revoke key → in-flight jobs fail after ~5 min cache; new jobs fail immediately
- Re-enable key → access resumes

Audit: AWS CloudTrail logs every Encrypt/Decrypt.

**Limits (live docs):** SFT supported today; DPO/RFT and final model-weight CMEK coming soon.

## Secure RFT workflow (BYOB + proprietary rewards)

1. Register BYOB dataset with rewards in your bucket
2. Keep reward/evaluator code in your environment (remote env or local)
3. Launch managed RFT with `firectl rftj` or eval-protocol, or Training API for full control
4. Revoke bucket IAM after job completes

Training API iterative RL: see cookbook examples and `references/rl-async.md` — do not paste long-lived cloud credentials in agent manifests.

## Related

- Surface choice + retention: live secure-fine-tuning docs
- Remote environments + tracing: `references/rft-agent-tracing.md`
- Inference ZDR: live data_handling docs
