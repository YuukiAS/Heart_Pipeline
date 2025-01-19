#!/bin/bash

# This file copies all the necessary file to a new folder so that it can be run on different sets of data, such as visit1 and visit2

set -e  # exit immediately when a command fails

ROOT_FOLDER="/work/users/y/u/yuukias/Heart_Pipeline"
TARGET_FOLDER="/overflow/htzhu/mingcheng/Heart_Pipeline"

start_time=$(date +%s)

if [ ! -d "$ROOT_FOLDER" ]; then
    echo "Error: ROOT_FOLDER does not exist: $ROOT_FOLDER"
    exit 1
fi

if [ -d "$TARGET_FOLDER" ]; then
    echo "TARGET_FOLDER exists. Deleting: $TARGET_FOLDER"
    rm -rf "$TARGET_FOLDER"
fi

echo "Creating TARGET_FOLDER: $TARGET_FOLDER"
mkdir -p "$TARGET_FOLDER"
mkdir -p "$TARGET_FOLDER"/src
mkdir -p "$TARGET_FOLDER"/data
mkdir -p "$TARGET_FOLDER"/code

echo "Copying README markdown file from $ROOT_FOLDER to $TARGET_FOLDER."
cp "$ROOT_FOLDER"/README.md "$TARGET_FOLDER/"

echo "Copying license file from $ROOT_FOLDER to $TARGET_FOLDER."
cp "$ROOT_FOLDER"/LICENSE "$TARGET_FOLDER/"

echo "Copying main-step scripts from $ROOT_FOLDER to $TARGET_FOLDER."
cp "$ROOT_FOLDER"/*.py "$TARGET_FOLDER/"

echo "Copying segmentation scripts from $ROOT_FOLDER to $TARGET_FOLDER."
cp -r "$ROOT_FOLDER"/src/segmentation "$TARGET_FOLDER"/src

echo "Copying feature extraction scripts from $ROOT_FOLDER to $TARGET_FOLDER."
cp -r "$ROOT_FOLDER"/src/feature_extraction "$TARGET_FOLDER"/src

echo "Copying trained model weights from $ROOT_FOLDER to $TARGET_FOLDER."
cp -r "$ROOT_FOLDER"/model "$TARGET_FOLDER/"
cp "$ROOT_FOLDER"/env_variable.sh "$TARGET_FOLDER/"

echo "Copying processing scripts from $ROOT_FOLDER to $TARGET_FOLDER."
cp -r "$ROOT_FOLDER"/script "$TARGET_FOLDER/"

echo "Copying utility scripts from $ROOT_FOLDER to $TARGET_FOLDER."
cp -r "$ROOT_FOLDER"/utils "$TARGET_FOLDER/"

echo "Copying validation scripts from $ROOT_FOLDER to $TARGET_FOLDER."
cp -r "$ROOT_FOLDER"/tasks "$TARGET_FOLDER/"

echo "Copying imported code from $ROOT_FOLDER to $TARGET_FOLDER."
cp -r "$ROOT_FOLDER"/lib "$TARGET_FOLDER/"

echo "Copying third-party softwares from $ROOT_FOLDER to $TARGET_FOLDER."
cp -r "$ROOT_FOLDER"/third_party "$TARGET_FOLDER/"

echo "Copying pipeline documentation from $ROOT_FOLDER to $TARGET_FOLDER."
cp -r "$ROOT_FOLDER"/doc "$TARGET_FOLDER/"

echo "Pipleine transferred successfully!"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total time taken: $elapsed_time seconds"