#!/bin/bash

path='RUN12_2'

# Find all .pdb files and process each file
find "$path" -type f -name '*.pdb' | while IFS= read -r file; do
    # Replace 'HIS' with 'HSI' in the file and overwrite the original file
    sed -i '' 's/HIS/HSI/g' "$file"
done




