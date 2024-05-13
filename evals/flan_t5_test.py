from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset
import sys
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/data/data")
parser.add_argument("--ekfac_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks")
parser.add_argument("--cov_batch_num", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/results")
parser.add_argument("--model_id", type=str, default="google/flan-t5-small")
parser.add_argument("--model_dir", type=str, default="/scratch/general/vast/u1420010/final_models/model")
parser.add_argument("--layers", nargs='+', type=str, default=['decoder.block.3.layer.2.DenseReluDense.wi_0', 'encoder.block.4.layer.1.DenseReluDense.wi_0'])

args = parser.parse_args()
sys.path.append(args.ekfac_dir)

from accelerate import Accelerator
from influence.base import KFACBaseInfluenceObjective, print_memory_usage
from influence.modules import EKFACInfluenceModule
import numpy as np
import torch

def main():
    accelerator = Accelerator()
    DEVICE = accelerator.device

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
            input_data = input_data.squeeze(0)
            label = self.tokenizer.encode(sample['choice'], return_tensors='pt')
            label = label[:, 0]
            return input_data, label
        
    def get_model_and_dataloader(data_path = args.data_dir+'/contract-nli/'):
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, truncation_side="right",  model_max_length=4200)

        dataset_train = CustomMNLIDataset(file_path=data_path+'T5_ready_train.json', tokenizer=tokenizer)
        train_dataloader = DataLoader(dataset_train, batch_size=1)

        dataset_dev = CustomMNLIDataset(file_path=data_path + 'T5_ready_dev.json', tokenizer=tokenizer)
        dev_dataloader = DataLoader(dataset_dev, batch_size=1)

        dataset_test = CustomMNLIDataset(file_path=data_path + 'T5_ready_test.json', tokenizer=tokenizer)
        test_dataloader = DataLoader(dataset_test, batch_size=1)

        return train_dataloader, dev_dataloader, test_dataloader

    train_loader, dev_dataloader, test_dataloader = get_model_and_dataloader()

    train_loader, dev_dataloader, test_dataloader = accelerator.prepare(train_loader, dev_dataloader, test_dataloader)
        
    class TransformerClassificationObjective(KFACBaseInfluenceObjective):
        def test_loss(self, model, batch):
            model_outputs = self.train_outputs(model, batch)
            output_probs = torch.log_softmax(model_outputs.logits, dim=-1)
            completion_probs = output_probs[:, -1]
            loss_fn = torch.nn.CrossEntropyLoss()
            # if batch[1].device != DEVICE:
            #     batch[1] = batch[1].to(DEVICE)
            return loss_fn(completion_probs, batch[1])
        
        def train_outputs(self, model, batch):
            # if next(model.parameters()).device != DEVICE:
            #     model = model.to(DEVICE)
            # if batch[0].device != DEVICE:
            #     batch[0] = batch[0].to(DEVICE)
            return model(input_ids=batch[0], decoder_input_ids=batch[0])
        
        def train_loss_on_outputs(self, outputs, batch):
            outputs = self.train_outputs(model, batch)
            # if batch[1].device != DEVICE:
            #     batch[1] = batch[1].to(DEVICE)
            loss_fn = torch.nn.CrossEntropyLoss()
            return loss_fn(outputs.logits[:, -1, :], batch[1].view(-1))

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

    module = EKFACInfluenceModule(
        model=model,
        objective=TransformerClassificationObjective(),
        train_loader=train_loader,
        test_loader=test_dataloader,
        device=DEVICE,
        accelerator=accelerator,
        layers=args.layers,
        n_samples=1
    )

    train_idxs = range(len(train_loader))
    test_idxs = range(len(test_dataloader))
    influences = module.influences(train_idxs, test_idxs)

    for layer in influences:
        with open(args.output_dir + f'/ekfac_influences_{layer}.txt', 'w') as f:
            for i, influence in enumerate(influences[layer]):
                f.write(f'{i}: {influence.tolist()}\n')
        f.close()

if __name__ == '__main__':
    main()
