#!/bin/bash
# Saga-based ticket booking orchestration script
# Coordinates: hold seats -> charge user -> confirm hold
# Usage: ./orchestrate.sh <user_id> <event_id> <seats> <total_price>

set -euo pipefail

RESTATE="http://localhost:8080"

USER_ID="${1:?Usage: $0 <user_id> <event_id> <seats> <total_price>}"
EVENT_ID="${2:?Usage: $0 <user_id> <event_id> <seats> <total_price>}"
SEATS="${3:?Usage: $0 <user_id> <event_id> <seats> <total_price>}"
PRICE="${4:?Usage: $0 <user_id> <event_id> <seats> <total_price>}"

echo "=== Booking: $SEATS seats for $EVENT_ID ==="
echo "  User: $USER_ID | Total price: $PRICE"
echo ""

# Step 1: Hold seats
echo "Step 1: Holding $SEATS seats..."
HOLD_RESP=$(curl -s -w "\n%{http_code}" \
  -X POST "$RESTATE/SeatInventory/$EVENT_ID/holdSeats" \
  -H "Content-Type: application/json" -d "$SEATS")
HOLD_CODE=$(echo "$HOLD_RESP" | tail -1)

if [ "$HOLD_CODE" -ne 200 ]; then
  echo "FAILED: Could not hold seats (HTTP $HOLD_CODE)"
  echo '{"status":"failed","reason":"hold_failed"}'
  exit 1
fi
echo "  Seats held."

# Step 2: Charge user
echo "Step 2: Charging user $PRICE..."
CHARGE_RESP=$(curl -s -w "\n%{http_code}" \
  -X POST "$RESTATE/UserAccount/$USER_ID/charge" \
  -H "Content-Type: application/json" -d "$PRICE")
CHARGE_BODY=$(echo "$CHARGE_RESP" | head -1)
CHARGE_CODE=$(echo "$CHARGE_RESP" | tail -1)

if [ "$CHARGE_BODY" != "true" ]; then
  echo "FAILED: Charge did not return expected response (got: $CHARGE_BODY)"
  echo '{"status":"failed","reason":"charge_failed"}'
  exit 1
fi
echo "  User charged."

# Step 3: Confirm hold
echo "Step 3: Confirming booking..."
CONFIRM_RESP=$(curl -s -w "\n%{http_code}" \
  -X POST "$RESTATE/SeatInventory/$EVENT_ID/confirmHold" \
  -H "Content-Type: application/json" -d "$SEATS")
CONFIRM_CODE=$(echo "$CONFIRM_RESP" | tail -1)

if [ "$CONFIRM_CODE" -ne 200 ]; then
  echo "FAILED: Could not confirm booking (HTTP $CONFIRM_CODE)"
  echo '{"status":"failed","reason":"confirm_failed"}'
  exit 1
fi

echo ""
echo "Booking confirmed."
echo '{"status":"success"}'
exit 0
