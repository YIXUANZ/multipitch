#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Define variables
MODEL_SCRIPT="path_to_your_model_script.py"  # Replace with the actual path to your Python script containing the Model class
TEST_AUDIO_FILE="path_to_your_audio_file.wav"  # Replace with the path to your test audio file
SP_MODEL_FILE="./pretrained/reverb_ref/sp_weights.pth"  # Path to speaker separation model weights
PT_MODEL_FILE="./pretrained/reverb_ref/pt_weights.pth"  # Path to pitch tracking model weights
OUTPUT_DIR="output"  # Directory to store the output
DEVICE="cuda:0"  # Specify the device (e.g., 'cuda:0' for GPU, 'cpu' for CPU)

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "Testing the model on audio file: $TEST_AUDIO_FILE"

# Run the model script
python3 $MODEL_SCRIPT --file $TEST_AUDIO_FILE --sp_model $SP_MODEL_FILE --pt_model $PT_MODEL_FILE --device $DEVICE --output $OUTPUT_DIR

echo "Test completed. Results saved in $OUTPUT_DIR."
