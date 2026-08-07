# Secure training operations

*Source of truth: live [Training overview](https://docs.fireworks.ai/fine-tuning/finetuning-intro.md), [Data handling](https://docs.fireworks.ai/guides/security_compliance/data_handling.md), and [Data security](https://docs.fireworks.ai/guides/security_compliance/data_security.md). Public policy, retention, and current product support stay in those docs.*

Use this reference for agent-assisted BYOB and CMEK setup. Always confirm the current cloud, method, and resource support in live docs before preparing commands.

## Bring your own bucket

BYOB lets Fireworks read a training dataset from customer-managed object storage. Use least-privilege access scoped to the required bucket and prefix, and revoke access after the approved workflow completes.

```bash
firectl dataset create <dataset-id> \
  --external-url gs://<bucket>/<prefix>/data.jsonl
```

Use the equivalent `s3://` or Azure URL only when the live docs confirm support for the selected workflow. Dataset registration is protected work and requires the final-plan confirmation in `SKILL.md`.

### GCS

Use the exact service-account identities supplied during onboarding. Grant only the documented bucket-policy read and object-view permissions required by the current flow. Do not copy service-account emails from another account or an old run.

### AWS S3

- Prefer OIDC federation to long-lived access keys.
- Restrict the trust policy by issuer, subject, and audience exactly as current onboarding specifies.
- Grant only `s3:GetObject` and the required `s3:ListBucket` prefix.
- Pass the reviewed IAM role only on commands that support it according to `--help`.

### Azure Blob

Prefer workload identity federation. If a supported workflow requires a Fireworks secret resource, reference its resource name without exposing the credential value.

## Customer-managed encryption keys

CMEK uses the customer's cloud KMS key to wrap Fireworks-managed data-encryption keys. Keep these concerns separate:

- BYOB controls where the source dataset lives.
- CMEK controls encryption of supported Fireworks-managed artifacts at rest.
- Neither substitutes for in-memory training isolation or inference data-handling policy.

Before setup:

1. Confirm the selected cloud and training method are supported in the live Training security section.
2. Confirm the exact issuer, subject, audience, and permissions for the user's account.
3. Show the role and key resource names without secret material.
4. Run the installed `firectl ... --help` for the current registration command.
5. Obtain confirmation before registering a key or starting a job.

Do not publish or automate a draft CLI contract from memory.

## Rotation and revocation

- Treat key disablement or IAM removal as a potentially disruptive action.
- Explain which active and stored artifacts depend on the key before asking for confirmation.
- Verify access after rotation with a read-only listing or status check.
- Use the cloud provider's audit log for KMS operations.

## Secure RFT

For proprietary prompts or rewards:

1. Keep dataset access least-privileged through BYOB when supported.
2. Keep evaluator or environment credentials in the secret-management surface.
3. Use a managed remote environment or Training API path appropriate to the reviewed workflow.
4. Follow `references/rft-agent-tracing.md` for redacted managed tracing or `references/rl-agentic.md` for custom Training API trajectories.
5. Revoke temporary storage access and tear down billable resources after the run.

## Non-negotiables

- Never place cloud credentials, signed URLs, raw IAM policy output, or customer data in a run report.
- Never reuse another account's onboarding identities or OIDC audience.
- Never claim a method or cloud is CMEK-supported without checking live docs.
- Keep human-readable policy in public docs; this reference only governs operational execution.

## Related

- Managed RFT: `managed-rft-operations.md`
- Managed remote tracing: `rft-agent-tracing.md`
- Training API agent trajectories: `rl-agentic.md`
- Failure handling: `error-reference.md`
