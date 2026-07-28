#!/bin/bash
# Saga-based ticket booking orchestration script (FIXED)
# Coordinates: hold seats -> charge user -> confirm hold
# With proper saga compensation on failure at each step.

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
HOLD_CODE=$(curl -s -o /tmp/hold_resp.txt -w "%{http_code}" \
  -X POST "$RESTATE/SeatInventory/$EVENT_ID/holdSeats" \
  -H "Content-Type: application/json" -d "$SEATS")

if [ "$HOLD_CODE" -ne 200 ]; then
  echo "FAILED: Could not hold seats (HTTP $HOLD_CODE)"
  echo '{"status":"failed","reason":"hold_failed"}'
  exit 1
fi
echo "  Seats held."

# Step 2: Charge user
# FIX: Check HTTP status code instead of response body content.
# The charge handler returns the remaining balance (a number), not a boolean.
echo "Step 2: Charging user $PRICE..."
CHARGE_CODE=$(curl -s -o /tmp/charge_resp.txt -w "%{http_code}" \
  -X POST "$RESTATE/UserAccount/$USER_ID/charge" \
  -H "Content-Type: application/json" -d "$PRICE")

if [ "$CHARGE_CODE" -ne 200 ]; then
  echo "FAILED: Charge failed (HTTP $CHARGE_CODE)"
  # FIX: Compensate by releasing held seats
  echo "  Compensating: releasing held seats..."
  curl -s -X POST "$RESTATE/SeatInventory/$EVENT_ID/releaseHold" \
    -H "Content-Type: application/json" -d "$SEATS" > /dev/null 2>&1 || true
  echo '{"status":"failed","reason":"charge_failed"}'
  exit 1
fi
echo "  User charged."

# Step 3: Confirm hold
echo "Step 3: Confirming booking..."
CONFIRM_CODE=$(curl -s -o /tmp/confirm_resp.txt -w "%{http_code}" \
  -X POST "$RESTATE/SeatInventory/$EVENT_ID/confirmHold" \
  -H "Content-Type: application/json" -d "$SEATS")

if [ "$CONFIRM_CODE" -ne 200 ]; then
  echo "FAILED: Could not confirm booking (HTTP $CONFIRM_CODE)"
  # FIX: Compensate by refunding the charge and releasing held seats
  echo "  Compensating: refunding charge and releasing seats..."
  curl -s -X POST "$RESTATE/UserAccount/$USER_ID/refund" \
    -H "Content-Type: application/json" -d "$PRICE" > /dev/null 2>&1 || true
  curl -s -X POST "$RESTATE/SeatInventory/$EVENT_ID/releaseHold" \
    -H "Content-Type: application/json" -d "$SEATS" > /dev/null 2>&1 || true
  echo '{"status":"failed","reason":"confirm_failed"}'
  exit 1
fi

echo ""
echo "Booking confirmed."
echo '{"status":"success"}'
exit 0
