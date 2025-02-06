#!/bin/bash

# This file copies all the necessary file to a new folder so that it can be run on different sets of data, such as visit1 and visit2

set -e # exit immediately when a command fails

SOURCE_FOLDER="/work/users/y/u/yuukias/Heart_Pipeline"
TARGET_FOLDER="/overflow/htzhu/mingcheng/Heart_Pipeline"

start_time=$(date +%s)

if [ ! -d "$SOURCE_FOLDER" ]; then
    echo "Error: SOURCE_FOLDER does not exist: $SOURCE_FOLDER"
    exit 1
fi

if [ -d "$TARGET_FOLDER" ]; then
    echo "TARGET_FOLDER exists. Deleting scripts: $TARGET_FOLDER"
    rm -rf "$TARGET_FOLDER"/src
    rm -rf "$TARGET_FOLDER"/code
    rm -rf "$TARGET_FOLDER"/utils
fi

echo "1/13: Creating TARGET_FOLDER: $TARGET_FOLDER"
mkdir -p "$TARGET_FOLDER"
mkdir -p "$TARGET_FOLDER"/data
mkdir -p "$TARGET_FOLDER"/src
mkdir -p "$TARGET_FOLDER"/code

echo "2/13: Copying README markdown file from $SOURCE_FOLDER to $TARGET_FOLDER."
cp -f "$SOURCE_FOLDER"/README.md "$TARGET_FOLDER/"

echo "3/13: Copying license file from $SOURCE_FOLDER to $TARGET_FOLDER."
cp -f "$SOURCE_FOLDER"/LICENSE "$TARGET_FOLDER/"

echo "4/13: Copying main-step scripts from $SOURCE_FOLDER to $TARGET_FOLDER."
cp -f "$SOURCE_FOLDER"/*.py "$TARGET_FOLDER/"

echo "5/13: Copying segmentation scripts from $SOURCE_FOLDER to $TARGET_FOLDER."
cp -rf "$SOURCE_FOLDER"/src/segmentation "$TARGET_FOLDER"/src

echo "6/13: Copying feature extraction scripts from $SOURCE_FOLDER to $TARGET_FOLDER."
cp -rf "$SOURCE_FOLDER"/src/feature_extraction "$TARGET_FOLDER"/src

echo "7/13: Copying trained model weights from $SOURCE_FOLDER to $TARGET_FOLDER."
cp -rf "$SOURCE_FOLDER"/model "$TARGET_FOLDER/"
cp -f "$SOURCE_FOLDER"/env_variable.sh "$TARGET_FOLDER/"

echo "8/13: Copying processing scripts from $SOURCE_FOLDER to $TARGET_FOLDER."
cp -rf "$SOURCE_FOLDER"/script "$TARGET_FOLDER/"

echo "9/13: Copying utility scripts from $SOURCE_FOLDER to $TARGET_FOLDER."
cp -rf "$SOURCE_FOLDER"/utils "$TARGET_FOLDER/"

echo "10/13: Copying validation scripts from $SOURCE_FOLDER to $TARGET_FOLDER."
cp -rf "$SOURCE_FOLDER"/tasks "$TARGET_FOLDER/"

echo "11/13: Copying imported code from $SOURCE_FOLDER to $TARGET_FOLDER."
cp -rf "$SOURCE_FOLDER"/lib "$TARGET_FOLDER/"

if [ -d "$TARGET_FOLDER"/third_party ]; then
    echo "12/13: Third-party softwares exist. No need to copy."
else
    echo "12/13: Copying third-party softwares from $SOURCE_FOLDER to $TARGET_FOLDER."
    cp -r "$SOURCE_FOLDER"/third_party "$TARGET_FOLDER/"
fi

if [ -d "$TARGET_FOLDER"/doc ]; then
    echo "13/13: Pipeline documentation exist. Only pages will be copied."
    cp -rf "$SOURCE_FOLDER"/doc/src "$TARGET_FOLDER/doc"
else
    echo "13/13: Copying pipeline documentation from $SOURCE_FOLDER to $TARGET_FOLDER."
    cp -rf "$SOURCE_FOLDER"/doc "$TARGET_FOLDER/"
fi

echo "Pipleine transferred successfully!"

echo "Don't forget to replace config.py with config_backup.py!!!"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total time taken: $elapsed_time seconds"