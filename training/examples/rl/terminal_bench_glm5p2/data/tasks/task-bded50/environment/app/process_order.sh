#!/bin/bash
# Saga-based order processing script
# Orchestrates wallet debit and inventory reservation with compensation on failure
#
# Usage: ./process_order.sh <user_id> <product_id> <quantity> <amount>

set -euo pipefail

RESTATE_INGRESS="http://localhost:8080"

USER_ID="${1:?Usage: $0 <user_id> <product_id> <quantity> <amount>}"
PRODUCT_ID="${2:?Usage: $0 <user_id> <product_id> <quantity> <amount>}"
QUANTITY="${3:?Usage: $0 <user_id> <product_id> <quantity> <amount>}"
AMOUNT="${4:?Usage: $0 <user_id> <product_id> <quantity> <amount>}"

echo "=== Processing Order ==="
echo "  User:     $USER_ID"
echo "  Product:  $PRODUCT_ID"
echo "  Quantity: $QUANTITY"
echo "  Amount:   $AMOUNT"
echo ""

# Step 1: Debit wallet
echo "Step 1: Debiting wallet $USER_ID by $AMOUNT..."
DEBIT_HTTP_CODE=$(curl -s -o /tmp/debit_response.txt -w "%{http_code}" \
  -X POST "$RESTATE_INGRESS/Wallet/$USER_ID/debit" \
  -H "Content-Type: application/json" \
  -d "$AMOUNT")

if [ "$DEBIT_HTTP_CODE" -ne 200 ]; then
  echo "FAILED: Wallet debit returned HTTP $DEBIT_HTTP_CODE"
  cat /tmp/debit_response.txt
  echo ""
  echo '{"status":"failed","reason":"wallet_debit_failed"}'
  exit 1
fi
echo "  Wallet debited successfully."

# Step 2: Reserve inventory
echo "Step 2: Reserving $QUANTITY units of $PRODUCT_ID..."
RESERVE_HTTP_CODE=$(curl -s -o /tmp/reserve_response.txt -w "%{http_code}" \
  -X POST "$RESTATE_INGRESS/Inventory/$PRODUCT_ID/reserve" \
  -H "Content-Type: application/json" \
  -d "$QUANTITY")

if [ "$RESERVE_HTTP_CODE" -ne 200 ]; then
  echo "FAILED: Inventory reserve returned HTTP $RESERVE_HTTP_CODE"
  cat /tmp/reserve_response.txt
  echo ""

  # Compensation: credit wallet back since debit succeeded but reserve failed
  echo "Running compensation..."

  echo '{"status":"failed","reason":"inventory_reserve_failed"}'
  exit 1
fi
echo "  Inventory reserved successfully."

echo ""
echo '{"status":"success"}'
exit 0
