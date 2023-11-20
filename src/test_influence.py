from model_eval import train_dataset
from influence_utils import EKFACInfluence
import torch
from torch.utils.data import Subset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import os


def main():

    pile_dataset = load_dataset("monology/pile-uncopyrighted", streaming=True)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m") 

    influence_model = EKFACInfluence(model, layers=['gpt_neox.layers.0.mlp.dense_h_to_4h'], influence_src_dataset=train_dataset, batch_size=128, cov_batch_size=128)

    test_dataset = Subset(train_dataset, range(500))
    test_dataset2 = Subset(test_dataset, range(500, 1000))
    influences = influence_model.influence(test_dataset)

if __name__ == '__main__':
    main()
