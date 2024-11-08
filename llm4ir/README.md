# ssl4llm

Directory are ignored:
data/ECG4/train.pt
data/ECG4/val.pt
data/ECG4/test.pt 

experiments_logs/...

The finetuned models' checkpoint are saved in experiments_logs.zip file

command to run the code:

'''
python main.py --experiment_description Etstcc --run_description run_1 --seed 0 --selected_dataset ECG4 --method TS-TCC


