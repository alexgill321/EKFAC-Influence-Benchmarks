from torch.nn.modules import Module
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset, Subset
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pile_dir", type=str, default="C:/Users/alexg/Documents/GitHub/pythia/data/")
parser.add_argument("--ekfac_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks")
parser.add_argument("--cov_batch_num", 
                    type=int, 
                    default=100,
                    help="Number of batches to compute activations for"
                    )
parser.add_argument("--output_dir", 
                    type=str, 
                    default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/results",
                    help="Output directory for results"
                    )
parser.add_argument("--model_id", 
                    type=str, 
                    default="EleutherAI/pythia-70m", 
                    help="Hugging Face model ID"
                    )
parser.add_argument("--layers", 
                    nargs='+', 
                    type=str, 
                    default=['gpt_neox.layers.1.mlp.dense_4h_to_h'], 
                    help="List of Layers to compute influence on"
                    )
args = parser.parse_args()
sys.path.append(args.ekfac_dir)

from influence.base import KFACBaseInfluenceObjective
from influence.modules import EKFACInfluenceModule
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.layers:
    print("Selected layers:", args.layers)
else:
    print("No layers selected.")

class PileDataset(Dataset):
    def __init__(self, indices):
        self.dataset = indices.tolist()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.dataset[idx])
        labels = torch.clone(input_ids)
        return input_ids, labels
    
data = np.load(args.pile_dir + "/indicies.npy", mmap_mode='r')
    
pile_dataset = PileDataset(data)

print("Dataset length:", len(pile_dataset))

pile_dataloader = DataLoader(pile_dataset, batch_size=1)
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

# for batch in pile_dataloader:
#     for i in batch:
#         print(tokenizer.decode(i))
#     break

model = AutoModelForCausalLM.from_pretrained(args.model_id)
print("Model loaded.")
model.to(DEVICE)
print("Model moved to device.")

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
        # Move model to DEVICE only if it's not already there to avoid unnecessary transfers.
        if next(model.parameters()).device != DEVICE:
            model = model.to(DEVICE)
        # Transfer the input batch to DEVICE only if it's not already there.
        # This check prevents redundant device transfers.
        if batch[0].device != DEVICE:
            batch[0] = batch[0].to(DEVICE)
        return model(batch[0])
    
    def train_loss_on_outputs(self, outputs, batch):
        outputs = self.train_outputs(model, batch)
        if batch[1].device != DEVICE:
            batch[1] = batch[1].to(DEVICE)
        labels_shift = batch[1][:, 1:]
        # Use swapaxes directly on logits without additional assignment to logits.
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        return loss_fn(outputs.logits.swapaxes(1, 2)[:, :, :-1], labels_shift)
    
    def pseudograd_loss(self, model, batch, n_samples=1, generator=None):
        with torch.no_grad():  # Context manager to temporarily disable gradient calculations
            outputs = self.train_outputs(model, batch)
            output_probs = torch.softmax(outputs.logits, dim=-1)
            samples = torch.multinomial(output_probs.view(-1, output_probs.size(-1)), num_samples=n_samples, replacement=True, generator=generator)
            sampled_labels = samples.view(outputs.logits.size(0), outputs.logits.size(1), n_samples)

        for s in range(n_samples):
            # Directly use the batch without cloning to save memory. Ensure batch is not modified in `train_loss_on_outputs`.
            sampled_batch = [batch[0], sampled_labels[:, :, s]]

            with torch.enable_grad():
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

# for batch in pile_dataloader:
#     model_outputs = model(batch[0])
#     break

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
    layers=args.layers,
    n_samples=1
)


train_idxs = range(0, len(pile_dataset))
test_idxs = [0, 1, 2, 3]
influences = module.influences(train_idxs=train_idxs, test_idxs=test_idxs)

for layer in influences:
    with open(args.output_dir +  f'/ekfac_influences_{layer}.txt', 'w') as f:
        for i, influence in enumerate(influences[layer]):
            f.write(f'{i}: {influence.tolist()}\n')
    f.close()


