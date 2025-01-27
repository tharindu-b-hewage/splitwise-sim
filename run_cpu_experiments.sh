#!/bin/bash

# Run the clean.sh script in the same folder
./clean.sh

# Define the types
#types=("code" "conv")
types=("conv")

# all
#rates=("30" "40" "50" "60" "70" "80" "90" "100" "110" "120" "130" "140" "150" "160" "170" "180" "190" "200" "210" "220" "230" "240" "250")
#rates=("30" "40" "50" "60" "70" "80" "90" "100" "110" "120" "130" "140" "150")
#rates=("130" "150" "170" "190" "200")
#rates=("30")
#rates=("30" "80" "130" "180" "230" "250")
rates=("40" "60" "80" "100")

# debug
#rates=("30")

# https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncadsh100v5-series?tabs=sizebasic
vm_types=("dgx-h100-with-cpu-vm40" "dgx-h100-with-cpu-vm80" "dgx-h100-with-cpu-vm112") # Standard_NC40ads_H100_v5, Standard_NC80ads_H100_v5, full machine

techniques=("linux" "least-aged" "proposed")

export HYDRA_FULL_ERROR=1

TEMP_RESULTS_FOLDER="results/0/splitwise_5_17"
FINAL_RESULTS_FOLDER="/Users/tharindu/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/phd-student/projects/dynamic-affinity/experiments"
BK_RESULTS_FOLDER="/Users/tharindu/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/phd-student/projects/dynamic-affinity/bk"

# Create a folder with the current date and time in the BK_RESULTS_FOLDER
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
BK_TIMESTAMPED_FOLDER="$BK_RESULTS_FOLDER/$TIMESTAMP"
mkdir -p "$BK_TIMESTAMPED_FOLDER"

# Move all folders in the FINAL_RESULTS_FOLDER to the new backup location
mv "$FINAL_RESULTS_FOLDER"/* "$BK_TIMESTAMPED_FOLDER"

for vm in "${vm_types[@]}"; do
  rm -rf 'cpu_core_frequencies.csv'
  #echo "vm_type: ""$vm"
  for technique in "${techniques[@]}"; do
    #echo "technique: ""$technique"
    sed -i '' "s/^task_allocation_algo=.*/task_allocation_algo=$technique/" cpu_configs.properties
    cat cpu_configs.properties
    for type in "${types[@]}"; do
      for rate in "${rates[@]}"; do
          #echo "--- type: $type with rate: $rate"
          echo "---------> " "$type" "trace at rate: " "$rate" "for" "$vm"
          sh scripts/run_splitwise_ha_cpu.sh "$type" "$rate" "$vm"
        done
      done
    results="$FINAL_RESULTS_FOLDER"/"$vm"/"$technique"
    mkdir -p "$results"
    mv "$TEMP_RESULTS_FOLDER"/* "$results"
  done
done