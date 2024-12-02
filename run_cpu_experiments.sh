#!/bin/bash

# Run the clean.sh script in the same folder
./clean.sh

# Define the types
types=("code" "conv")

# all
#rates=("30" "40" "50" "60" "70" "80" "90" "100" "110" "120" "130" "140" "150" "160" "170" "180" "190" "200" "210" "220" "230" "240" "250")
#rates=("30" "40" "50" "60" "70" "80" "90" "100" "110" "120" "130" "140" "150")

# debug
rates=("30")

# Loop over each type
for type in "${types[@]}"; do
  # Loop over rates from 30 to 250 with an interval of 10
for rate in "${rates[@]}"; do
    # Your commands go here. For example:
    echo "Experiment: type: $type with rate: $rate"
    sh scripts/run_splitwise_ha_cpu.sh "$type" "$rate"
  done
done