from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import psutil

# pile_dataset = load_dataset("monology/pile-uncopyrighted", streaming=True)

# print(next(iter(pile_dataset["train"])))

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

for mod in model.modules():
    print(mod)
