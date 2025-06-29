#!/bin/bash

# Read the dataset_path from config.yaml
echo "Reading dataset_path from config.yaml..."
dataset_path=$(grep '^[[:space:]]*dataset_path:' config.yaml | grep -v '^[[:space:]]*#' | awk '{print $2}')
echo "Dataset path: $dataset_path"

# Check if the dataset_path exists, if not, create it
if [ ! -d "$dataset_path" ]; then
    echo "Dataset path does not exist. Creating directory: $dataset_path"
    mkdir -p "$dataset_path"
else
    echo "Dataset path already exists: $dataset_path"
fi

# Create the required subdirectories
echo "Creating subdirectory: $dataset_path/dats"
mkdir -p "$dataset_path/dats"

echo "Creating subdirectory: $dataset_path/features"
mkdir -p "$dataset_path/features"

echo "Creating subdirectory: $dataset_path/location"
mkdir -p "$dataset_path/location"

echo "Creating subdirectory: $dataset_path/vision"
mkdir -p "$dataset_path/vision"

echo "Folders created successfully in $dataset_path"