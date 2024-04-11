import ast
from torch.utils.data import Dataset
import argparse
import json
import pandas as pd
import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/data/data")
parser.add_argument("--influence_scores_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/results/flan-t5-xl")
parser.add_argument("--output_dir", type=str, default="C:/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks/results")

args = parser.parse_args()

class CustomMNLITruncDataset(Dataset):
    def __init__(self, file_path):
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
        label = sample['true_label']
        return input_data, label
    
class CustomMNLIDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)['data']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_data = sample['input']
        label = sample['choice']
        return input_data, label
    
def get_dataloaders(data_path = args.data_dir+'/contract-nli/'):

    dataset_train = CustomMNLIDataset(file_path=data_path+'T5_ready_train.json')

    dataset_test = CustomMNLITruncDataset(file_path=data_path + 'Influential_split_full.csv')

    return dataset_train, dataset_test

def main():

    train_dataset, test_dataset = get_dataloaders()
    mean_influence = None
    n_layers = 24
    for i in tqdm.tqdm(range(n_layers)):
        for block_type in ['encoder', 'decoder']:
            if block_type == 'encoder':
                filename = args.influence_scores_dir + f"/ekfac_influences_encoder.block.{i}.layer.1.DenseReluDense.wo_full-split.txt"
            else:
                filename = args.influence_scores_dir + f"/ekfac_influences_decoder.block.{i}.layer.2.DenseReluDense.wo_full-split.txt"

            influences = torch.tensor([])
            with open(filename, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split(':')
                    influence = torch.tensor(ast.literal_eval(parts[1]))
                    if len(influences) == 0:
                        influences = influence.view(1, -1)
                    else:
                        influences = torch.cat([influences, influence.view(1, -1)])
            if mean_influence is None:
                mean_influence = influences
            else:
                mean_influence.add_(influences)

    influence_scores = torch.div(mean_influence, n_layers*2)
    top_scores, top_indices = torch.topk(influence_scores, 3, dim=1)

    df = pd.DataFrame(columns=["input", "label", "top1", "top2", "top3"])
    for i, row in enumerate(top_indices):
        row = row.tolist()
        query, label = test_dataset[i]
        df = df.append({"input": query, "label": label, "top1": train_dataset[row[0]][0], "top2": train_dataset[row[1]][0], "top3": train_dataset[row[2]][0]}, ignore_index=True)

    df.to_csv(args.output_dir + '/k_top_scores.csv', index=False)
        

        
if __name__ == "__main__":
    main()