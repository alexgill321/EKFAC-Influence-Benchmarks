from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset, Subset
import sys
import argparse
import json
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/data/data")
parser.add_argument("--ekfac_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks")
parser.add_argument("--cov_batch_num", type=int, default=10)
parser.add_argument("--output_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/results")
parser.add_argument("--model_dir", type=str, default="google/flan-t5-small")
parser.add_argument("--layers", nargs='+', type=str, default='all')
parser.add_argument("--test_size", type=int, default=None)
parser.add_argument("--model_max_len", type=int, default=3000)
parser.add_argument("--svd", type=bool, default=True)
args = parser.parse_args()
sys.path.append(args.ekfac_dir)

from influence import KFACBaseInfluenceObjective, EKFACInfluenceModule  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("num devices: ", torch.cuda.device_count())

model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
model.to(DEVICE)

class CustomMNLITruncDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        df = pd.read_csv(file_path)
        # We create two separate records for each row: one for left and one for right truncates, both sharing the same label
        left_truncates = df[['input_left_truncate', 'true_label']].rename(columns={'input_left_truncate': 'input'})
        right_truncates = df[['input_right_truncate', 'true_label']].rename(columns={'input_right_truncate': 'input'})
        # Combine these records into a single dataframe
        self.data_frame = pd.concat([left_truncates, right_truncates], ignore_index=True)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx]
        input_data = sample['input']
        input_data = self.tokenizer.encode(input_data, return_tensors='pt', truncation= True)
        input_data = input_data.squeeze(0).to(DEVICE)
        label = sample['true_label']
        label = self.tokenizer.encode(label, return_tensors='pt')
        label = label[:, 0].to(DEVICE)
        return input_data, label
    
class CustomMNLIDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        with open(file_path, 'r') as f:
            self.data = json.load(f)['data']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_data = sample['input']
        input_data = self.tokenizer.encode(input_data, return_tensors='pt', truncation= True)
        input_data = input_data.squeeze(0).to(DEVICE)
        label = self.tokenizer.encode(sample['choice'], return_tensors='pt')
        label = label[:, 0].to(DEVICE)
        return input_data, label
    
def get_dataloaders(data_path = args.data_dir+'/contract-nli/'):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,truncation_side="right",  model_max_length=args.model_max_len)

    dataset_train = CustomMNLIDataset(file_path=data_path+'T5_ready_train.json', tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset_train, batch_size=1)

    dataset_dev = CustomMNLIDataset(file_path=data_path + 'T5_ready_dev.json', tokenizer=tokenizer)
    dev_dataloader = DataLoader(dataset_dev, batch_size=1)

    dataset_test = CustomMNLITruncDataset(file_path=data_path + 'Influential_split_full.csv', tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset_test, batch_size=1)

    if args.cov_batch_num is not None:
        cov_dataset = Subset(dataset_train, range(args.cov_batch_num))
        cov_dataloader = DataLoader(cov_dataset, batch_size=1)
    else:
        cov_dataloader = train_dataloader

    return train_dataloader, dev_dataloader, test_dataloader, cov_dataloader

train_loader, dev_dataloader, test_dataloader, cov_dataloader = get_dataloaders()

print(len(test_dataloader))

class TransformerClassificationObjective(KFACBaseInfluenceObjective):
    def test_loss(self, model, batch):
        outputs = self.train_outputs(model, batch)
        # if batch[1].device != DEVICE:
        #     batch[1] = batch[1].to(DEVICE)
        return outputs.loss
    
    def train_outputs(self, model, batch):
        # if next(model.parameters()).device != DEVICE:
        #     model = model.to(DEVICE)
        # if batch[0].device != DEVICE:
        #     batch[0] = batch[0].to(DEVICE)
        return model(input_ids=batch[0], labels=batch[1])
    
    def train_loss_on_outputs(self, outputs, batch):
        output = self.train_outputs(model, batch)
        # if batch[1].device != DEVICE:
        #     batch[1] = batch[1].to(DEVICE)
        
        return output.loss

    def pseudograd_loss(self, model, batch, n_samples=1, generator=None):
        with torch.no_grad():  # Context manager to temporarily disable gradient calculations
            outputs = self.train_outputs(model, batch)
            output_probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
            samples = torch.multinomial(output_probs.view(-1, output_probs.size(-1)), num_samples=n_samples, replacement=True, generator=generator)
            sampled_labels = samples.view(outputs.logits.size(0), 1, n_samples)

        for s in range(n_samples):
            sampled_batch = [batch[0], sampled_labels[:,:,s]]

            with torch.enable_grad():
                yield self.train_loss_on_outputs(outputs, sampled_batch)

if args.layers == 'all':
    layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear) and name.endswith('.wo'):
            layers.append(name)
else:
    layers = args.layers

module = EKFACInfluenceModule(
    model=model,
    objective=TransformerClassificationObjective(),
    train_loader=train_loader,
    test_loader=test_dataloader,
    cov_loader=cov_dataloader,
    device=DEVICE,
    layers=layers,
    n_samples=1
)

train_idxs = range(len(train_loader))

if args.test_size is None:
    args.test_size = len(test_dataloader)
    num_full_batches = 1
    remainder = 0
else:
    args.test_size = min(args.test_size, len(test_dataloader))  
    num_full_batches = len(test_dataloader) // args.test_size
    remainder = len(test_dataloader) % args.test_size

for batch_idx in range(num_full_batches):
    start_idx = batch_idx * args.test_size
    end_idx = (batch_idx + 1) * args.test_size
    print(f"Batch {batch_idx}: {start_idx} - {end_idx}")
    test_idxs = range(start_idx, end_idx)
    influences = module.influences(train_idxs, test_idxs, args.svd)

    for layer in influences:
        output_file_path = f"{args.output_dir}/ekfac_influences_{layer}_full-split.txt"
        with open(output_file_path, 'w') as f:
            for idx, influence in enumerate(influences[layer]):
                f.write(f'{idx}: {influence.tolist()}\n')

# Handle the last batch if there's a remainder
if remainder > 0:
    start_idx = num_full_batches * args.test_size
    end_idx = start_idx + remainder
    test_idxs = range(start_idx, end_idx)
    influences = module.influences(train_idxs, test_idxs, args.svd)

    for layer in influences:
        output_file_path = f"{args.output_dir}/ekfac_influences_{layer}_full-split.txt"
        with open(output_file_path, 'w') as f:
            for idx, influence in enumerate(influences[layer]):
                f.write(f'{idx}: {influence.tolist()}\n')