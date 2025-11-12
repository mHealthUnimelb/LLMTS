#!/bin/bash

for i in {1..3}
do
    target_dir="./data/P12_${i}"

    # create the directory if it doesn't exist
    if [ ! -d "$target_dir" ]; then
        echo "Folder '$target_dir' does not exist. Creating it..."
        mkdir -p "$target_dir"
    else
        echo "Folder '$target_dir' already exists."
    fi

    cd "$target_dir" || exit
    echo "Enterd $(pwd)"

    python ../../get_data.py \
        --classif \
        --data_split_path ../../../datasets/P12data/splits/phy12_split${i}.npy \
        --batch_size 128 \
        --quantization 0.016 \
        --out_dir ./ \
        --base_name P12
    
    # Go back to the previous directory
    cd - >/dev/null
    
    echo "Finished run for P12_${i}"
    echo "---------------------------------------"
done

echo "All runs completed!"
