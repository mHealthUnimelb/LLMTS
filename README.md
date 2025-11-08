# CALF
## ENV:
Install python 3.9 and then run the following command:
```
cd CALF
pip install -r requirements.txt
```

## RUN:
1. Run `cd CALF`.

2. Run `python pca.py` to get `wte_pca_600.pt`.

3. All scripts are located in `scripts` folder, you can run like `bash ./scripts/train_P12.sh`.


# FSCA
## ENV:


## RUN:
All scripts are located in `FSCA/Classification/scripts` folder, you can run like 
```
cd FSCA/Classification
bash ./scripts/train_P12_1.sh
```


# MOMENT
## ENV:


## RUN:
All scripts are located in `moment/scripts` folder, you can run like 
```
cd moment
bash ./scripts/MIMIC_1.sh
```


# mTAND:
## ENV:
It uses the same environment as CALF.

## RUN:
All scripts are located in `mTAND/scripts` folder, you can run like 
```
cd mTAND
bash ./scripts/P12.sh
```


# S2IP-LLM
## ENV:


## RUN:
All scripts are located in `S2IP-LLM/Irregular_Classification/scripts` and `S2IP-LLM/Regular_Classification/scripts` folders, you can run like 
```
cd S2IP-LLM/Irregular_Classification
bash ./scripts/P12_1.sh
```
```
cd Regular_Classification
bash ./scripts/ECG.sh
```


# Time-LLM:
## ENV:


## RUN:
All prompts are located in `Time-LLM/dataset/prompt_bank`.

All scripts are located in `Time-LLM/scripts` folder, you can run like `
```
cd Time-LLM
bash ./scripts/TimeLLM_ECG.sh
```


# UniTS
## ENV:


## RUN:
You can download datasets from ... and put them in folder `UniTS/data`, like 
```
--UniTS
    -- data
        -- MIMIC
            -- MIMIC_1
            -- MIMIC_2
            -- MIMIC_3
        -- P12_1
        -- P12_2
        -- P12_3
```
You can run shell scripts (P12_1.sh, bash P12_2.sh, bash P12_3.sh, MIMIC_1.sh, MIMIC_2.sh, MIMIC_3.sh) like 
```
cd UniTS
bash P12_1.sh
```


# Waprformer
## ENV:


## RUN:
All scripts are located in `Waprformer/scripts` folder, you can run like
```
cd Waprformer
bash scripts/{script}.sh
```