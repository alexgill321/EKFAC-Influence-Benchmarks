from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset, Subset
import sys
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/data/data")
parser.add_argument("--ekfac_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks")
parser.add_argument("--cov_batch_num", type=int, default=10)
parser.add_argument("--output_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/results")
parser.add_argument("--model_dir", type=str, default="google/flan-t5-small")
parser.add_argument("--layers", nargs='+', type=str, default='all')
parser.add_argument("--test_start_idx", type=int, default=0)
parser.add_argument("--test_end_idx", type=int, default=1000)
args = parser.parse_args()
sys.path.append(args.ekfac_dir)

from influence.base import KFACBaseInfluenceObjective
from influence.modules import EKFACInfluenceModule
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
model.to(DEVICE)

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
    
def get_model_and_dataloader(data_path = args.data_dir+'/contract-nli/'):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, truncation_side="right",  model_max_length=3000)

    dataset_train = CustomMNLIDataset(file_path=data_path+'T5_ready_train.json', tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset_train, batch_size=1)

    dataset_dev = CustomMNLIDataset(file_path=data_path + 'T5_ready_dev.json', tokenizer=tokenizer)
    dev_dataloader = DataLoader(dataset_dev, batch_size=1)

    dataset_test = CustomMNLIDataset(file_path=data_path + 'T5_ready_test.json', tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset_test, batch_size=1)

    if args.cov_batch_num is not None:
        cov_dataset = Subset(dataset_train, range(args.cov_batch_num))
        cov_dataloader = DataLoader(cov_dataset, batch_size=1)
    else:
        cov_dataloader = train_dataloader    

    return train_dataloader, dev_dataloader, test_dataloader, cov_dataloader

train_loader, dev_dataloader, test_dataloader, cov_dataloader = get_model_and_dataloader()

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
test_idxs = range(args.test_start_idx, args.test_end_idx)
influences = module.influences(train_idxs, test_idxs)

for layer in influences:
    with open(args.output_dir + f'/ekfac_influences_{layer}.txt', 'w') as f:
        for i, influence in enumerate(influences[layer]):
            f.write(f'{i}: {influence.tolist()}\n')
    f.close()
