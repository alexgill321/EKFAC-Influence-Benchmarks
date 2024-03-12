from torch.nn.modules import Module
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset, Subset
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pile_dir", type=str, default="C:/Users/alexg/Documents/GitHub/pythia/data/")
parser.add_argument("--ekfac_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks")
parser.add_argument("--cov_batch_num", type=int, default=3)
args = parser.parse_args()
sys.path.append(args.ekfac_dir)

from influence.base import KFACBaseInfluenceObjective
from influence.modules import EKFACInfluenceModule
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

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
    
data = np.load(args.pile_dir + "/indicies.npy", mmap_mode='r')

print("loaded data")
    
pile_dataset = PileDataset(data)

pile_dataloader = DataLoader(pile_dataset, batch_size=1)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

# for batch in pile_dataloader:
#     for i in batch:
#         print(tokenizer.decode(i))
#     break

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
model.to(DEVICE)

class PileObjective(KFACBaseInfluenceObjective):
    def test_loss(self, model, batch):
        inputs = torch.concat([batch[0], batch[1]], dim=1)
        model_outputs = model(inputs)
        output_probs = torch.log_softmax(model_outputs.logits, dim=-1)
        completion_probs = output_probs[:, batch[0].size(1)-1:-1]
        prob = torch.tensor(0.0).to(DEVICE)
        for i in range(completion_probs.size(1)):
            prob.add_(completion_probs[0, i, batch[1][0, i]])
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
    
queries = [("How are you today?", "I would like to destroy the universe."),
           ("I Must Not Fear."," Fear Is The Mind-Killer. Fear Is The Little Death That Brings Obliteration."),
           ("television rules the nation", "around the world"),
           ("what is the best thing that has ever been created?", " shrek the third of course.")]
    
tokenized_prompts = [(tokenizer(prompt, return_tensors="pt").input_ids.squeeze(dim=0).to(DEVICE), 
                      tokenizer(completion, return_tensors="pt").input_ids.squeeze(dim=0).to(DEVICE)) for (prompt, completion) in queries]

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

for batch in pile_dataloader:
    model_outputs = model(batch[0])
    break

# for batch in prompt_dataloader:
#     inputs = torch.concat([batch[0], batch[1]], dim=1)
#     model_outputs = model(inputs)
#     output_probs = torch.log_softmax(model_outputs.logits, dim=-1)
#     completion_probs = output_probs[:, batch[0].size(1)-1:-1]
#     prob = torch.tensor(0.0).to(DEVICE)

#     for i in range(completion_probs.size(1)):
#         prob.add_(completion_probs[0, i, batch[1][0, i]])
#     break


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
