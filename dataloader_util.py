import os
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def load_data_from_path(dir_path= '/scratch/general/vast/u1420010/final_models/data/contract-nli/'):

    directory_path = Path(dir_path)
    
    if directory_path.exists():
        train_ds = load_dataset('json', data_files=str(directory_path)+'/T5_ready_train.json', field = 'data', split="train")
        val_ds = load_dataset('json', data_files=str(directory_path)+'/T5_ready_dev.json', field = 'data', split="train")
        test_ds = load_dataset('json', data_files=str(directory_path)+'/T5_ready_test.json', field = 'data', split="train")
         

        print(type(test_ds))

    else:
        print("this path does not exsist")
        return _, _, _

    return train_ds, val_ds, test_ds




def format_dataset(examples):
    inputs = tokenizer.batch_encode_plus(examples['input'], truncation=False)
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
    }




def get_model_and_dataloader():    
    train_ds, val_ds, test_ds = load_data_from_path()
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", trust_remote_code=True)






    tokenizer.pad_token = tokenizer.eos_token


    tokenized_dataset_train = train_ds.map(format_dataset,
                                    batched=True)
    #tokenized_dataset_train = tokenized_dataset_train.remove_columns(train_ds.column_names)


    print(type(tokenized_dataset_train))
    #data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #tokenized_dataset_train = tokenized_dataset_train['input_ids']
    train_loader = DataLoader(tokenized_dataset_train, batch_size=1, shuffle=True)
    itr = 0
    for data_batch in train_loader:
    

        itr+=1
        #tokens = tokenizer.decode([x.item() for x in data_batch['input_ids']])
    

        input_ids = data_batch['input_ids']
        input_ids = torch.tensor([x.item() for x in data_batch['input_ids']]).to(device)

        break
    


    print(itr)