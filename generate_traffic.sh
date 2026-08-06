#!/bin/bash

echo "Starting randomized loan propensity traffic generator..."
echo "Targets: http://localhost:8000/score"
echo "Press Ctrl+C at any time to stop the stream."
echo "--------------------------------------------------"

while true; do
  # --- 1. GENERATE RANDOMIZED VALUES ---
  ID="CUST-$((RANDOM % 90000 + 10000))"
  AGE=$((RANDOM % 53 + 18))                  # 18 to 70
  INCOME=$(awk -v r=$RANDOM 'BEGIN {printf "%.2f", 15000 + (r % 135000)}') # 15k to 150k
  GENDER_ENC=$((RANDOM % 2))                 # 0 or 1
  
  TXN_COUNT=$((RANDOM % 81 + 5))             # 5 to 85 total transactions
  INFLOW_COUNT=$((RANDOM % 16 + 2))          # 2 to 17 deposits
  OUTFLOW_COUNT=$((TXN_COUNT - INFLOW_COUNT)) # Rest are withdrawals
  
  TOTAL_INFLOW=$(awk -v r=$RANDOM 'BEGIN {printf "%.2f", 2000 + (r % 10000)}')
  TOTAL_OUTFLOW=$(awk -v r=$RANDOM -v inf=$TOTAL_INFLOW 'BEGIN {printf "%.2f", (inf * 0.7) + (r % 2000)}')
  NET_CASHFLOW=$(awk -v inf=$TOTAL_INFLOW -v out=$TOTAL_OUTFLOW 'BEGIN {printf "%.2f", inf - out}')
  
  AVG_BALANCE=$(awk -v r=$RANDOM 'BEGIN {printf "%.2f", 500 + (r % 15000)}')
  MIN_BALANCE=$(awk -v avg=$AVG_BALANCE -v r=$RANDOM 'BEGIN {printf "%.2f", avg * 0.2 + (r % 200)}')
  STD_BALANCE=$(awk -v avg=$AVG_BALANCE 'BEGIN {printf "%.2f", avg * 0.15}')
  
  BAL_TO_INC=$(awk -v avg=$AVG_BALANCE -v inc=$INCOME 'BEGIN {printf "%.4f", avg / (inc / 12)}')
  TXN_PER_MONTH=$(awk -v tx=$TXN_COUNT 'BEGIN {printf "%.1f", tx / 3.0}') 
  
  TOTAL_LOANS=$((RANDOM % 4))                # 0 to 3 loans
  if [ $TOTAL_LOANS -gt 0 ]; then
    NUM_DECLINED=$((RANDOM % 2))             # 0 or 1 declined
    MAX_LOAN=$(awk -v r=$RANDOM 'BEGIN {printf "%.2f", 1000 + (r % 24000)}')
    MEAN_LOAN=$(awk -v max=$MAX_LOAN 'BEGIN {printf "%.2f", max * 0.75}')
  else
    NUM_DECLINED=0
    MAX_LOAN="0.0"
    MEAN_LOAN="0.0"
  fi

  # --- 2. SEND PAYLOAD TO API ---
  STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/score \
    -H "Content-Type: application/json" \
    -d "[{
      \"ID\": \"$ID\",
      \"AGE\": $AGE,
      \"INCOME\": $INCOME,
      \"GENDER_ENC\": $GENDER_ENC,
      \"txn_count\": $TXN_COUNT,
      \"inflow_count\": $INFLOW_COUNT,
      \"outflow_count\": $OUTFLOW_COUNT,
      \"total_inflow\": $TOTAL_INFLOW,
      \"total_outflow\": $TOTAL_OUTFLOW,
      \"net_cashflow\": $NET_CASHFLOW,
      \"avg_balance\": $AVG_BALANCE,
      \"min_balance\": $MIN_BALANCE,
      \"std_balance\": $STD_BALANCE,
      \"balance_to_income_ratio\": $BAL_TO_INC,
      \"txn_per_month\": $TXN_PER_MONTH,
      \"total_loans\": $TOTAL_LOANS,
      \"num_declined\": $NUM_DECLINED,
      \"max_loan_amount\": $MAX_LOAN,
      \"mean_loan_amount\": $MEAN_LOAN
    }]")

  echo "Customer: $ID | Age: $AGE | Income: \$$INCOME | API Response: $STATUS_CODE"
  sleep 1
done
