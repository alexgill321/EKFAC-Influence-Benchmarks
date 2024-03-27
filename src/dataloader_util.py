import json
import os
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding

from torch.utils.data import DataLoader, Dataset
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



class CustomMNLIDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.choices_mapping = {"yes": 0, "Yes": 0, "entailment": 0, "Entailment": 0,
                           "cannot say": 1, "Cannot say": 1, "can't say": 1, "Can't say": 1,
                           "not enough information": 1, "NotMentioned": 1,
                           "no": 2, "No": 2, "contradiction": 2, "Contradiction": 2}
        self.tokenizer = tokenizer
        with open(file_path, 'r') as f:
            self.data = json.load(f)['data']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_data = sample['input']
        input_data = self.tokenizer.encode(input_data, return_tensors='pt', truncation= True).to(device)
        label = sample['choice']


        return input_data, self.choices_mapping[label]


def get_model_and_dataloader(data_path = '/scratch/general/vast/u1420010/final_models/data/contract-nli/'):


    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl", truncation_side="right",  model_max_length=4200)

    dataset_train = CustomMNLIDataset(file_path=data_path+'T5_ready_train.json',
                                tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset_train, batch_size=1)

    dataset_dev = CustomMNLIDataset(file_path=data_path + 'T5_ready_dev.json',
                                      tokenizer=tokenizer)
    dev_dataloader = DataLoader(dataset_dev, batch_size=1)

    dataset_test = CustomMNLIDataset(file_path=data_path + 'T5_ready_test.json',
                                      tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset_test, batch_size=1)

    return train_dataloader, dev_dataloader, test_dataloader



train_loader, dev_dataloader, test_dataloader = get_model_and_dataloader()

for batch in dev_dataloader:
    print(batch[0].shape)
    break



