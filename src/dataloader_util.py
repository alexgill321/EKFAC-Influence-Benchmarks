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


class ContractMNLIDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data_batch = self.dataset[idx]
        input_ids = torch.tensor([x.item() for x in data_batch['input_ids']]).to(device)

        return input_ids, data_batch['choice']


def get_model_and_dataloader():    
    train_ds, val_ds, test_ds = load_data_from_path()
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", trust_remote_code=True)


    def format_dataset(examples):
        inputs = tokenizer.batch_encode_plus(examples['input'], truncation=False)
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
        }
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset_train = train_ds.map(format_dataset,
                                    batched=True)
    
    tokenized_dataset_val = val_ds.map(format_dataset, batched = True)

    tokenized_dataset_test = test_ds.map(format_dataset, batched = True)
    #tokenized_dataset_train = tokenized_dataset_train.remove_columns(train_ds.column_names)

    train_ds = ContractMNLIDataset(tokenized_dataset_train)
    val_ds = ContractMNLIDataset(tokenized_dataset_val)
    test_ds = ContractMNLIDataset(tokenized_dataset_test)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = get_model_and_dataloader()

for batch in train_loader:
    print(batch)

