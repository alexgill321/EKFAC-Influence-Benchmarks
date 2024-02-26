from torch.nn.modules import Module
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pile_dir", type=str, default="C:/Users/alexg/Documents/GitHub/pythia/data/")
parser.add_argument("--ekfac_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/")
parser.add_argument("--cov_batch_num", type=int, default=3)
args = parser.parse_args()
sys.path.append(args.ekfac_dir)

from influence.base import KFACBaseInfluenceObjective
from influence.modules import EKFACInfluenceModule
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PileDataset(Dataset):
    def __init__(self, indices):
        self.dataset = indices.tolist()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.dataset[idx]).to(DEVICE)
        labels = torch.cat([input_ids[1:], input_ids[:1]], dim=0)

        labels = torch.clone(input_ids)
        return input_ids, labels
    
data = np.load(args.pile_dir + "indicies.npy")
    
pile_dataset = PileDataset(data)

pile_dataloader = DataLoader(pile_dataset, batch_size=1)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

# for batch in pile_dataloader:
#     for i in batch:
#         print(tokenizer.decode(i))
#     break

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
model.to(DEVICE)

for batch in pile_dataloader:
    outputs = model(batch[0], labels=batch[1])
    labels = batch[1]
    labels_shift = labels[:, 1:]
    logits = outputs.logits.swapaxes(1, 2)[:, :, :-1]
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    print(loss_fn(logits, labels_shift))
    break
 
inputs = tokenizer("tell me a joke", return_tensors="pt").input_ids.to(DEVICE)
output = model.generate(inputs, max_length=20, output_scores=True, return_dict_in_generate=True)

class PileObjective(KFACBaseInfluenceObjective):
    def test_loss(self, model, batch):
        outputs = model.generate(batch, max_length=100, output_scores=True, return_dict_in_generate=True)
        log_probs = torch.log_softmax(outputs, dim=2)
        prob = torch.sum(torch.max(log_probs, dim=2), dim=1)
        return prob
    
    def train_outputs(self, model, batch):
        model = model.to(DEVICE)
        batch[0] = batch[0].to(DEVICE)
        return model(batch[0])
    
    def train_loss_on_outputs(self, outputs, batch):
        outputs = self.train_outputs(model, batch)
        batch[1] = batch[1].to(DEVICE)
        labels_shift = batch[1][:, 1:]
        logits = outputs.logits.swapaxes(1, 2)[:, :, :-1]
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        return loss_fn(logits, labels_shift)
    
    def pseudograd_loss(self, model, batch, n_samples=1, generator=None):
        outputs = self.train_outputs(model, batch)
        output_probs = torch.softmax(outputs.logits, dim=-1)
        outputs_2d = output_probs.reshape(-1, output_probs.size(-1))
        samples = torch.multinomial(outputs_2d, num_samples=n_samples, replacement=True, generator=generator)
        sampled_labels = samples.view(outputs.logits.size(0), outputs.logits.size(1), n_samples)
        for s in range(n_samples):
            s = sampled_labels[:, :, s]
            inputs = batch[0].clone()
            sampled_batch = [inputs, s]
            yield self.train_loss_on_outputs(outputs, sampled_batch)
    
prompts = ["Hello, world!", "How are you doing?", "This is an example prompt."]    
tokenized_prompts = [tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE) for prompt in prompts]

class PromptDataset(Dataset):
    def __init__(self, tokenized_prompts):
        self.dataset = tokenized_prompts
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

prompt_dataset = PromptDataset(tokenized_prompts)
prompt_subset = Subset(pile_dataset, indices=range(args.cov_batch_num))
cov_dataloader = DataLoader(prompt_subset, batch_size=1)
prompt_dataloader = DataLoader(prompt_dataset, batch_size=1)

module = EKFACInfluenceModule(
    model=model,
    objective=PileObjective(),
    train_loader=pile_dataloader,
    test_loader=prompt_dataloader,
    cov_loader=cov_dataloader,
    device=DEVICE,
    layers=['gpt_neox.layers.1.mlp.dense_4h_to_h'],
    n_samples=1
)

train_idxs = range(0, 100)
test_idxs = [0]
influences = module.influences(train_idxs=train_idxs, test_idxs=test_idxs)
