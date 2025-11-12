#!/bin/bash

seeds=(12 123 42)

for i in {1..3}
do
    seed=${seeds[$((i-1))]}

    target_dir="./data/MIMIC/mimic_classification/MIMIC_${i}"

    # create the directory if it doesn't exist
    if [ ! -d "$target_dir" ]; then
        echo "Folder '$target_dir' does not exist. Creating it..."
        mkdir -p "$target_dir"
    else
        echo "Folder '$target_dir' already exists."
    fi

    cd "$target_dir" || exit
    echo "Enterd $(pwd)"

    python ../../../../get_data.py \
        --seed $seed \
        --classif \
        --dataset MIMIC \
        --batch_size 128 \
        --out_dir ./ \
        --base_name MIMIC
    
    # Go back to the previous directory
    cd - >/dev/null
    
    echo "Finished run for MIMIC_${i} (seed=${seed})"
    echo "---------------------------------------"
done

echo "All runs completed!"

