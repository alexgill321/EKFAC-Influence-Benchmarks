import os
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set the random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings)

def prepare_dataset(dataset_name = '', tokenizer_name = ''):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    train_ds, val_ds = load_custom_huggingface_dataset(nrows = '')

    train_ds = train_ds.select(range(47))
    val_ds = val_ds.select(range(47))


    print("starting tokenization")

    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset_train = train_ds.map(lambda examples: tokenizer(examples["text"],  truncation=True, padding='longest'),
                                    batched=True)
    tokenized_dataset_val = val_ds.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding='longest'),
        batched=True)

    tokenized_dataset_train = tokenized_dataset_train['input_ids']
    tokenized_dataset_val = tokenized_dataset_val['input_ids']



    #
    # custom_train_dataobj = CustomDataset(tokenized_dataset_train)
    # custom_test_dataobj = CustomDataset(tokenized_dataset_val)

    train_loader = DataLoader(tokenized_dataset_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(tokenized_dataset_val, batch_size=1, shuffle=True)
    print("created dataloaders")

    #
    # for x in train_loader:
    #     print(len(x), (x[0].shape))
    #     exit()

    # train_loader = DataLoader(train_ds['train'], batch_size=16, shuffle=True)
    # test_loader = DataLoader(val_ds['train'], batch_size=16, shuffle=True)

    return train_loader, test_loader


def get_model():
    # Instantiate the model, loss function, and optimizer
    model = HuggingFaceModelWrapper()

    ####changed the reduction here from none to mean
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    #mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.model.parameters(), lr=0.0001)
    return model, optimizer

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class HuggingFaceModelWrapper(nn.Module):
    def __init__(self, model_name="roneneldan/TinyStories-1M"):
        super(HuggingFaceModelWrapper, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids):

        outputs = self.model(input_ids)
        return outputs.logits
        # inputs = self.tokenizer(text, return_tensors="pt", return_attention_mask=False)
        # outputs = self.model.model.generate(inputs['input_ids'], 250)
        # return outputs
        #
        # encodings = self.tokenizer(text, return_tensors="pt")
        # input_ids, attention_mask = encodings.input_ids, encodings.attention_mask



    def generate(self, input_ids):
        labels = input_ids.clone()
        labels[:, :-1] = labels[:, 1:].clone()
        labels[:, -1] = -100



        print(f"labels ={labels}")

        outputs = self.model.generate(input_ids = input_ids, output_scores = True, max_length=50, return_dict_in_generate=False)
        # all_completion_scores = []
        #
        # for completion_score in outputs.scores:
        #     print(completion_score.shape)
        #     softmax_completion_scores = self.softmax(completion_score)
        #     print(softmax_completion_scores.shape)
        #     x = torch.max(softmax_completion_scores, dim =1 )
        #     all_completion_scores.append(x)
        # print(all_completion_scores[0].shape)
        # exit()
        # print(outputs.keys())
        # exit()
        return outputs


def load_custom_huggingface_dataset(store_data='TinyStories', dataset_name='roneneldan/TinyStories', nrows=''):
    ### we will load a dataset from huggingface, defaults to tiny stories

    directory_path = Path(os.getcwd() + '/data/' + store_data)
    huggingface_ds = ''

    if directory_path.exists():
        train_ds = load_dataset('json', data_files=str(directory_path)+'/train.json', split="train")
        val_ds = load_dataset('json', data_files=str(directory_path)+'/validation.json', split="train")
    # first time loading dataset
    else:
        huggingface_ds = load_dataset(dataset_name)
        train_ds = huggingface_ds['train']
        val_ds = huggingface_ds['validation']
        for split, split_dataset in huggingface_ds.items():
            print(os.getcwd() + '/data/' + f"{dataset_name.split('/')[-1]}/{split}.json")
            split_dataset.to_json(os.getcwd() + '/data/' + f"{dataset_name.split('/')[-1]}/{split}.json")


    return train_ds, val_ds
