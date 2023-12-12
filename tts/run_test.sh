#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found, please install Python3."
    exit
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "pip3 could not be found, please install pip3."
    exit
fi

# Install required Python libraries
echo "Installing required Python libraries..."
pip3 install transformers torch TTS

# Define arguments for the Python script
instances_arr=(1)  # Example: number of instances
inferences_arr=(25)  # Example: number of inferences
model_names=("VITS")  # Example: model names

# Run the Python script with different arguments
for model in "${model_names[@]}"
do
    for instances in "${instances_arr[@]}"
    do
        for inferences in "${inferences_arr[@]}"
        do
            echo "Running with $instances instances, $inferences inferences, model $model"
            python3 test.py --instances "$instances" --inferences "$inferences" --model "$model"
        done
    done
done
