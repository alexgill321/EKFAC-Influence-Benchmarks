import argparse
import logging
from kronfluence.task import Task
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on Pythia model.")

    parser.add_argument(
        "--model_id",
        type=str,
        default="EleutherAI/pythia-70m"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="C:/Users/alexg/Documents/GitHub/pythia/data/"
    )

    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=32,
        help="Rank for the low-rank query gradient approximation.",
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/results"
    )

    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=16,
        help="Batch size for computing query gradients.",
    )

    parser.add_argument(
        "--use_half_precision",
        type=bool,
        default=False,
        help="Whether to use half precision for computing factors and scores.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size for computing training gradients.",
    )

    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )

    parser.add_argument(
        "--cov_batch_num",
        type=int,
        default=100,
        help="Number of batches to compute activations for"
    )

    args = parser.parse_args()

    return args

class PileDataset(Dataset):
    def __init__(self, indices):
        self.dataset = indices.tolist()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.dataset[idx])
        labels = torch.clone(input_ids)
        return input_ids, labels

class PromptDataset(Dataset):
    def __init__(self, tokenized_prompts):
        self.dataset = tokenized_prompts
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)

    data = np.load(args.data_dir + "/indicies.npy", mmap_mode='r')

    pile_dataset = PileDataset(data)

    queries = [("How are you today?", "I would like to destroy the universe."),
           ("I Must Not Fear."," Fear Is The Mind-Killer. Fear Is The Little Death That Brings Obliteration."),
           ("television rules the nation", "around the world"),
           ("what is the best thing that has ever been created?", " shrek the third of course.")]
    
    tokenized_prompts = [(tokenizer(prompt, return_tensors="pt").input_ids.squeeze(dim=0), 
                          tokenizer(completion, return_tensors="pt").input_ids.squeeze(dim=0)) for (prompt, completion) in queries]
    
    prompt_dataset = PromptDataset(tokenized_prompts)

    class PythiaTask(Task):
        def compute_train_loss(self, batch, model, sample = False):
            outputs = model(batch[0])
            loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
            if not sample:
                labels_shift = batch[1][:, 1:]
                return loss_fn(outputs.logits.swapaxes(1, 2)[:, :, :-1], labels_shift)
            else:
                with torch.no_grad():  
                    output_probs = torch.softmax(outputs.logits, dim=-1)
                    samples = torch.multinomial(output_probs.view(-1, output_probs.size(-1)), num_samples=1, replacement=True)
                    sampled_labels = samples.view(outputs.logits.size(0), outputs.logits.size(1))
                    sampled_labels = sampled_labels[:, 1:]
                return loss_fn(outputs.logits.swapaxes(1, 2)[:, :, :-1], sampled_labels)
                
        def compute_measurement(self, batch, model):
            return self.compute_train_loss(batch, model)

        def tracked_modules(self):
            total_modules = []
            
            for name, layer in model.named_modules():
                if isinstance(layer, torch.nn.Linear) and (name.endswith('.dense_h_to_4h') or name.endswith('.dense_4h_to_h')):
                    total_modules.append(name)

            return total_modules
        
    task = PythiaTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="pythia-kronfluence-test",
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
        dataset=Subset(pile_dataset, list(range(args.train_batch_size * args.cov_batch_num))),
        per_device_batch_size=None,
        initial_per_device_batch_size_attempt=args.train_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )

    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    score_args = ScoreArguments(
        query_gradient_rank=rank,
        query_gradient_svd_dtype=torch.float32
    )

    scores_name = f'{factor_args.strategy}_pythia_scores'
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
        factors_name=factors_name,
        query_dataset=prompt_dataset,
        query_indices=list(range(len(prompt_dataset))),
        train_dataset=pile_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )

    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")

if __name__ == "__main__":
    main()
