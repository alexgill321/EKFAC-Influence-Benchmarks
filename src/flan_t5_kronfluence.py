import argparse
import json
import pandas as pd
import logging
from kronfluence.task import Task
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments

import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on Flan-T5-xl model.")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="google/flan-t5-small"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/data/data"
    )

    parser.add_argument(
        "--model_max_len",
        type=int,
        default=4200,
        help="Max sequence length of model inputs."
    )

    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=32,
        help="Rank for the low-rank query gradient approximation.",
    )

    parser.add_argument(
        "--use_half_precision",
        type=bool,
        default=False,
        help="Whether to use half precision for computing factors and scores.",
    )

    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=1,
        help="Batch size for computing query gradients.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )

    args = parser.parse_args()

    return args

class FlanT5Task(Task):
    def compute_train_loss(self, batch, model, sample = False):
        outputs = model(input_ids = batch[0], labels = batch[1])

        if not sample:
            return outputs.loss
        else:
            with torch.no_grad():
                probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
                sample = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
                sampled_labels = sample.view(outputs.logits.size(0), 1)
            
            with torch.enable_grad():
                return model(input_ids = batch[0], labels = sampled_labels).loss
            
    def compute_measurement(self, batch, model):
        return self.compute_train_loss(batch, model)

    def tracked_modules(self):
        total_modules = []
        
        for i in range(8):
            total_modules.append(f'decoder.block.{i}.layer.2.DenseReluDense.wo')
            total_modules.append(f'encoder.block.{i}.layer.1.DenseReluDense.wo')

        return total_modules
    
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
        input_data = input_data.squeeze(0)
        label = sample['true_label']
        label = self.tokenizer.encode(label, return_tensors='pt')
        label = label[:, 0]
        return input_data, label


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    tokenizer=AutoTokenizer.from_pretrained(args.model_dir, truncation_side="right",  model_max_length=args.model_max_len)

    train_dataset = CustomMNLIDataset(file_path=args.data_dir+'/contract-nli/T5_ready_train.json', tokenizer=tokenizer)

    eval_dataset = CustomMNLITruncDataset(file_path=args.data_dir+'/contract-nli/Influential_split_full.csv', tokenizer=tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    print(model)

    task = FlanT5Task()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="flan_t5_kron",
        model=model,
        task=task,
        profile=True
    )

    factors_name = args.factor_strategy
    factor_args = FactorArguments(strategy=args.factor_strategy)
    if args.use_half_precision:
        factor_args.activation_covariance_dtype = torch.bfloat16
        factor_args.gradient_covariance_dtype = torch.bfloat16
        factor_args.lambda_dtype = torch.bfloat16
        factors_name += "_half"

    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=Subset(train_dataset, list(range(300))),
        per_device_batch_size=1,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )

    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    score_args = ScoreArguments(
        query_gradient_rank=rank,
        query_gradient_svd_dtype=torch.float32
    )
    scores_name = f'{factor_args.strategy}_flan_t5_scores'
    if rank is not None:
        scores_name += f'_rank{rank}'

    if args.use_half_precision:
        score_args.per_sample_gradient_dtype = torch.bfloat16
        score_args.score_dtype = torch.bfloat16
        score_args.cached_activation_cpu_offload = True
        scores_name += "_half"

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=args.factor_strategy,
        query_dataset=eval_dataset,
        query_indices=list(range(len(eval_dataset))),
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )

    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]

    logging.info(f"Scores shape: {scores.shape}")

if __name__ == "__main__":
    main()
