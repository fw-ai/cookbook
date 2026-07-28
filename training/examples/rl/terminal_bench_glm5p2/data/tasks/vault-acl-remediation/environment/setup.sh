#!/bin/bash
set -e

export VAULT_ADDR="http://127.0.0.1:8200"

# Start Vault server in background
vault server -config=/app/vault-config.hcl &>/var/log/vault/server.log &
VAULT_PID=$!

# Wait for Vault to be ready
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8200/v1/sys/health?standbyok=true\&sealedcode=200\&uninitcode=200 >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Initialize Vault with a single unseal key
INIT_OUTPUT=$(vault operator init -key-shares=1 -key-threshold=1 -format=json)
UNSEAL_KEY=$(echo "$INIT_OUTPUT" | jq -r '.unseal_keys_b64[0]')
ROOT_TOKEN=$(echo "$INIT_OUTPUT" | jq -r '.root_token')

vault operator unseal "$UNSEAL_KEY" >/dev/null

echo "$ROOT_TOKEN" > /app/root_token.txt
echo "$UNSEAL_KEY" > /app/unseal_key.txt

export VAULT_TOKEN="$ROOT_TOKEN"

# Wait for active status
for i in $(seq 1 15); do
    if vault status -format=json 2>/dev/null | jq -e '.sealed == false' >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# ================================================================
# SECRETS ENGINES
# ================================================================

# KV v2 — max_versions left at default 0 (unlimited)
vault secrets enable -path=secret kv-v2 >/dev/null

# Populate multi-team secrets
vault kv put secret/app-team/database host=db.internal port=5432 password=app_db_s3cret >/dev/null
vault kv put secret/app-team/api-keys stripe=sk_live_abc123 sendgrid=SG.xyz789 >/dev/null
vault kv put secret/data-team/warehouse host=warehouse.internal port=5439 password=dw_s3cret >/dev/null
vault kv put secret/data-team/ml-config model_key=mk_prod_456 endpoint=https://ml.internal >/dev/null
vault kv put secret/platform/tls-certs cert_path=/etc/ssl/cert.pem key_path=/etc/ssl/key.pem >/dev/null
vault kv put secret/platform/monitoring grafana_token=gt_prod_789 pagerduty_key=pd_xxx >/dev/null

# PKI — Root CA
vault secrets enable pki >/dev/null
vault secrets tune -max-lease-ttl=87600h pki >/dev/null
vault write -format=json pki/root/generate/internal \
    common_name="Example Root CA" \
    ttl=87600h >/dev/null
vault write pki/config/urls \
    issuing_certificates="http://127.0.0.1:8200/v1/pki/ca" \
    crl_distribution_points="http://127.0.0.1:8200/v1/pki/crl" >/dev/null

# PKI Intermediate CA
vault secrets enable -path=pki_int pki >/dev/null
vault secrets tune -max-lease-ttl=43800h pki_int >/dev/null

INT_CSR=$(vault write -format=json pki_int/intermediate/generate/internal \
    common_name="Example Intermediate CA" | jq -r '.data.csr')
echo "$INT_CSR" > /tmp/int.csr

SIGNED_CERT=$(vault write -format=json pki/root/sign-intermediate \
    csr=@/tmp/int.csr format=pem_bundle ttl=43800h | jq -r '.data.certificate')
echo "$SIGNED_CERT" > /tmp/int_signed.pem

vault write pki_int/intermediate/set-signed certificate=@/tmp/int_signed.pem >/dev/null

# PKI role — MISCONFIGURED: allows any domain name
vault write pki_int/roles/web-certs \
    allow_any_name=true \
    max_ttl=720h \
    key_type=rsa \
    key_bits=2048 \
    enforce_hostnames=true >/dev/null

# PKI role — correctly configured (red herring)
vault write pki_int/roles/internal-certs \
    allowed_domains="internal.example.com" \
    allow_subdomains=true \
    allow_any_name=false \
    max_ttl=168h >/dev/null

# Transit engine with multiple keys
vault secrets enable transit >/dev/null

# MISCONFIGURED: weak cipher (aes128 instead of aes256)
vault write -f transit/keys/payments-key type=aes128-gcm96 >/dev/null

# MISCONFIGURED: exportable key — once set, cannot be changed back
vault write -f transit/keys/backup-key type=aes256-gcm96 exportable=true allow_plaintext_backup=true >/dev/null

# Correctly configured key (red herring)
vault write -f transit/keys/signing-key type=ed25519 >/dev/null

# ================================================================
# POLICIES
# ================================================================

# app-team policy — correctly scoped (not a finding)
vault policy write app-team - <<'POLICY'
path "secret/data/app-team/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "secret/metadata/app-team/*" {
  capabilities = ["read", "list"]
}
path "pki_int/issue/web-certs" {
  capabilities = ["create", "update"]
}
path "transit/encrypt/payments-key" {
  capabilities = ["create", "update"]
}
path "transit/decrypt/payments-key" {
  capabilities = ["create", "update"]
}
POLICY

# MISCONFIGURED: data-team policy uses + segment wildcard, granting
# access to ALL teams' secret paths (secret/data/ANY_TEAM/*)
vault policy write data-team - <<'POLICY'
path "secret/data/+/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "secret/metadata/+/*" {
  capabilities = ["read", "list"]
}
path "transit/encrypt/payments-key" {
  capabilities = ["create", "update"]
}
POLICY

# platform-admin policy — broad but intentional for ops
vault policy write platform-admin - <<'POLICY'
path "sys/health" {
  capabilities = ["read"]
}
path "sys/mounts" {
  capabilities = ["read", "list"]
}
path "sys/mounts/*" {
  capabilities = ["create", "read", "update", "delete"]
}
path "sys/auth" {
  capabilities = ["read", "list"]
}
path "sys/auth/*" {
  capabilities = ["create", "read", "update", "delete", "sudo"]
}
path "sys/policies/acl/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "sys/audit" {
  capabilities = ["read", "list", "sudo"]
}
path "sys/audit/*" {
  capabilities = ["create", "read", "update", "delete", "sudo"]
}
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "pki/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}
path "pki_int/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}
path "transit/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
POLICY

# ================================================================
# AUTH METHODS
# ================================================================

# Userpass auth — TTLs set excessively high
vault auth enable userpass >/dev/null

vault write auth/userpass/users/app-service \
    password=app_pass_2024 \
    token_ttl=768h \
    token_max_ttl=768h \
    policies=app-team >/dev/null

vault write auth/userpass/users/data-analyst \
    password=data_pass_2024 \
    token_ttl=768h \
    token_max_ttl=768h \
    policies=data-team >/dev/null

vault write auth/userpass/users/platform-ops \
    password=ops_pass_2024 \
    token_ttl=24h \
    token_max_ttl=72h \
    policies=platform-admin >/dev/null

# AppRole auth
vault auth enable approle >/dev/null

# Legitimate AppRole for CI/CD — correctly configured
vault write auth/approle/role/ci-deploy \
    token_policies="app-team" \
    token_ttl=1h \
    token_max_ttl=4h \
    secret_id_ttl=24h \
    secret_id_num_uses=1 \
    bind_secret_id=true >/dev/null

# MISCONFIGURED: legacy role with no secret binding + admin privileges
# Use the HTTP API directly to ensure proper JSON types for the
# boolean and array fields (token_bound_cidrs must be a JSON array).
curl -sf --request POST \
    --header "X-Vault-Token: $VAULT_TOKEN" \
    --data '{
        "token_policies": ["platform-admin"],
        "token_ttl": "168h",
        "token_max_ttl": "720h",
        "bind_secret_id": false,
        "secret_id_num_uses": 0,
        "token_bound_cidrs": ["0.0.0.0/0"]
    }' \
    "$VAULT_ADDR/v1/auth/approle/role/legacy-migration" >/dev/null

# ================================================================
# FAKE AUDIT LOG — planted on disk to mislead
# No actual Vault audit device is enabled
# ================================================================
mkdir -p /var/log/vault
cat > /var/log/vault/audit.log <<'AUDIT_EOF'
{"time":"2024-11-15T10:23:45.123Z","type":"request","auth":{"client_token":"hmac-sha256:a1b2c3","accessor":"hmac-sha256:d4e5f6","display_name":"userpass-data-analyst","policies":["data-team","default"],"token_policies":["data-team","default"]},"request":{"id":"req-7f3a1","operation":"read","path":"secret/data/app-team/database","remote_address":"10.0.1.50"}}
{"time":"2024-11-15T10:23:45.456Z","type":"response","auth":{"client_token":"hmac-sha256:a1b2c3","accessor":"hmac-sha256:d4e5f6","display_name":"userpass-data-analyst","policies":["data-team","default"],"token_policies":["data-team","default"]},"request":{"id":"req-7f3a1","operation":"read","path":"secret/data/app-team/database"},"response":{"data":{"data":"hmac-sha256:x7y8z9"}}}
{"time":"2024-11-15T10:24:12.789Z","type":"request","auth":{"client_token":"hmac-sha256:a1b2c3","accessor":"hmac-sha256:d4e5f6","display_name":"userpass-data-analyst","policies":["data-team","default"],"token_policies":["data-team","default"]},"request":{"id":"req-b2c4e","operation":"read","path":"secret/data/app-team/api-keys","remote_address":"10.0.1.50"}}
{"time":"2024-11-15T10:24:12.901Z","type":"response","auth":{"client_token":"hmac-sha256:a1b2c3","accessor":"hmac-sha256:d4e5f6","display_name":"userpass-data-analyst","policies":["data-team","default"],"token_policies":["data-team","default"]},"request":{"id":"req-b2c4e","operation":"read","path":"secret/data/app-team/api-keys"},"response":{"data":{"data":"hmac-sha256:q1r2s3"}}}
{"time":"2024-11-15T10:30:01.234Z","type":"request","auth":{"client_token":"hmac-sha256:a1b2c3","accessor":"hmac-sha256:d4e5f6","display_name":"userpass-data-analyst","policies":["data-team","default"],"token_policies":["data-team","default"]},"request":{"id":"req-m8n9p","operation":"read","path":"secret/data/platform/monitoring","remote_address":"10.0.1.50"}}
{"time":"2024-11-15T10:30:01.567Z","type":"response","auth":{"client_token":"hmac-sha256:a1b2c3","accessor":"hmac-sha256:d4e5f6","display_name":"userpass-data-analyst","policies":["data-team","default"],"token_policies":["data-team","default"]},"request":{"id":"req-m8n9p","operation":"read","path":"secret/data/platform/monitoring"},"response":{"data":{"data":"hmac-sha256:t4u5v6"}}}
AUDIT_EOF

echo "Multi-tenant Vault deployment bootstrapped."
echo "Root token: /app/root_token.txt"
echo "Unseal key: /app/unseal_key.txt"
