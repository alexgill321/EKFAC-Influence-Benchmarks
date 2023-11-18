from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import psutil

pile_dataset = load_dataset("monology/pile-uncopyrighted", streaming=True)

pile_loader = DataLoader(pile_dataset["train"], batch_size=2)

data_iter = iter(pile_loader)
print(next(data_iter))

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

for mod in model.modules():
    print(mod)
