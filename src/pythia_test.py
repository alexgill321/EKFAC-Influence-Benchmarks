from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

class PileDataset(Dataset):
    def __init__(self, indices):
        self.dataset = indices.tolist()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx])
    
pile_dataset = PileDataset(np.load('C:/Users/alexg/Documents/GitHub/pythia/data/indicies.npy'))

pile_dataloader = DataLoader(pile_dataset, batch_size=2)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")


# for batch in pile_dataloader:
#     for i in batch:
#         print(tokenizer.decode(i))
#     break

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

for name, mod in model.named_modules():
    print(name)
