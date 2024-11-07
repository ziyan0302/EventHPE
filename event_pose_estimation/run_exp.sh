#!/bin/bash

# Specify the Python command and script path
PYTHON_COMMAND="/home/ziyan/anaconda3/envs/eventhpeVenv/bin/python"
SCRIPT_PATH="/home/ziyan/02_research/EventHPE/event_pose_estimation/train-eventcap-v2.py"
TEST_PATH="/home/ziyan/02_research/EventHPE/event_pose_estimation/test-eventcap-v2.py"
# Specify the output file
OUTPUT_FILE="output.txt"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Define an array of argument sets
    # "--batch_optimization_epochs 0 --event_refinement_epochs 200 --stab_loss 0"
    # "--batch_optimization_epochs 0 --event_refinement_epochs 200"
    # "--batch_optimization_epochs 0 --event_refinement_epochs 200 --sil_loss 500 --stab_loss 50"
    # "--batch_optimization_epochs 2000 --event_refinement_epochs 0"
    # "--batch_optimization_epochs 2000 --event_refinement_epochs 100"
    # "--batch_optimization_epochs 2500 --event_refinement_epochs 100"
    # "--batch_optimization_epochs 3000 --event_refinement_epochs 100"
    # "--batch_optimization_epochs 2000 --event_refinement_epochs 100 --lr_event 0.005"
declare -a ARGUMENT_SETS=(
    "--batch_optimization_epochs 0 --event_refinement_epochs 0"
    "--batch_optimization_epochs 0 --event_refinement_epochs 20"
    "--batch_optimization_epochs 2000 --event_refinement_epochs 30"
)

# Loop through each set of arguments and run the Python script
for ((i=0; i<${#ARGUMENT_SETS[@]}; i++))
do
    echo "Run #$((i+1)) with args: ${ARGUMENT_SETS[i]}" >> "$OUTPUT_FILE"  # Log the argument set
    $PYTHON_COMMAND $SCRIPT_PATH ${ARGUMENT_SETS[i]} >> "$OUTPUT_FILE" 2>&1  # Run the script with args and append output
    echo "-----------------" >> "$OUTPUT_FILE"  # Separate runs in the output file
    $PYTHON_COMMAND $TEST_PATH >> "$OUTPUT_FILE" 2>&1
    echo "=================" >> "$OUTPUT_FILE"  # Separate runs in the output file

done

echo "All runs completed. Output saved to $OUTPUT_FILE"
