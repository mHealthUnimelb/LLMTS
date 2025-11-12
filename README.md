# Datasets
## ECG:
Run the following command:
```
cd datasets
bash generate_ECG.sh
```

## PhysioNet 2012:
1. Run the following command:
    ```
    cd datasets
    python physionet.py
    ```
2. Download P12data from https://doi.org/10.6084/m9.figshare.19514341.v1. Aftering downloading and extracting, place `P12data` folder in the `datasets` folder.


## MIMIC-III:
Follow the preprocessing of t-PatchGNN(https://github.com/usail-hkust/t-PatchGNN) and GRU-ODE(https://github.com/edebrouwer/gru_ode_bayes).


# Model
## CALF
### ENV:
Install python 3.9 and then run the following command:
```
cd CALF
pip install -r requirements.txt
```

### RUN:
1. Run `cd CALF`.

2. Run `python pca.py` to get `wte_pca_600.pt`.

3. All scripts are located in `scripts` folder, you can run like `bash ./scripts/train_P12.sh`.


## FSCA
### ENV:
Install python 3.8 and run the following command:
```
cd FSCA
pip install -r requirements.txt
```

### RUN:
All scripts are located in `FSCA/Classification/scripts` folder, you can run like 
```
cd FSCA/Classification
bash ./scripts/train_P12_1.sh
```


## MOMENT
### ENV:
Install python 3.11 and run the following command:
```
cd moment
pip install -r requirements.txt
```

### RUN:
All scripts are located in `moment/scripts` folder, you can run like 
```
cd moment
bash ./scripts/MIMIC_1.sh
```


## mTAND:
### ENV:
It uses the same environment as CALF.

### RUN:
All scripts are located in `mTAND/scripts` folder, you can run like 
```
cd mTAND
bash ./scripts/P12.sh
```


## S2IP-LLM
### ENV:
Install python 3.10 and run the following command:
```
cd S2IP-LLM
pip install -r requirements.txt
```

### RUN:
All scripts are located in `S2IP-LLM/Irregular_Classification/scripts` and `S2IP-LLM/Regular_Classification/scripts` folders, you can run like 
```
cd S2IP-LLM/Irregular_Classification
bash ./scripts/P12_1.sh
```
```
cd Regular_Classification
bash ./scripts/ECG.sh
```


## Time-LLM:
### ENV:
Install python 3.11 and run the following command:
```
cd Time-LLM
pip install -r requirements.txt
```

### RUN:
All prompts are located in `Time-LLM/dataset/prompt_bank`.

All scripts are located in `Time-LLM/scripts` folder, you can run like `
```
cd Time-LLM
bash ./scripts/TimeLLM_ECG.sh
```


## UniTS
### ENV:
Install python 3.9.21 and run the following command:
```
cd UniTS
pip install -r requirements.txt
```


### RUN:
1. Since UniTS requires data to be in .ts format, the first step is to convert the data to .ts format. You can do this by running the following command:
    ```
    cd UniTS
    bash convert_P12_to_ts.sh
    bash convert_MIMIC_to_ts.sh
    ```
    It should be noted that due to the large size of the MIMIC dataset, it requires 100GB of memory to run for me.

    After the above command is completed, the structure of the `UniTS/data` is as follows:
    ```
    --UniTS
        -- data
            -- MIMIC
                -- mimic_classification
                    -- MIMIC_1
                        -- MIMIC_TEST.ts
                        -- MIMIC_TRAIN.ts
                    -- MIMIC_2
                        -- ...
                    -- MIMIC_3
                        -- ...
            -- P12_1
                -- P12_TEST.ts
                -- P12_TRAIN.ts
            -- P12_2
                -- ...
            -- P12_3
                -- ...
    ```

2. Download pretrained weights `units_x128_pretrain_checkpoint.pth` from [UniTS Pretrained weights](https://github.com/mims-harvard/UniTS/releases/tag/ckpt) and place it in the `UniTS` folder.

3. You can run shell scripts (P12_1.sh, bash P12_2.sh, bash P12_3.sh, MIMIC_1.sh, MIMIC_2.sh, MIMIC_3.sh) like 
    ```
    cd UniTS
    bash P12_1.sh
    ```


## Waprformer
### ENV:
Install python 3.7.16 and run the following command:
```
cd Waprformer
pip install -r requirements.txt
```

### RUN:
All scripts are located in `Waprformer/scripts` folder, you can run like
```
cd Waprformer
bash scripts/{script}.sh
```


# Acknowledgements

Our gratitude extends to the authors of the following repositories for their model implementations:

- [CALF](https://github.com/Hank0626/CALF)
- [FSCA](https://github.com/tokaka22/ICLR25-FSCA)
- [MOMENT](https://github.com/moment-timeseries-foundation-model/moment)
- [mTAND](https://github.com/reml-lab/mTAN)
- [S2IP-LLM](https://github.com/panzijie825/s2ip-llm)
- [Time-LLM](https://github.com/KimMeen/Time-LLM)
- [UniTS](https://github.com/mims-harvard/UniTS)
- [Warpformer](https://github.com/imJiawen/Warpformer)
