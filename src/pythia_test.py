from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

pile_dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
print(pile_dataset)


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
tokenized_data = pile_dataset.map(lambda x: tokenizer(x['text'], return_tensors="pt", padding=True), batched=True, batch_size=5)
input_dataloader = DataLoader(tokenized_data, batch_size=2)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(**inputs)

for batch in input_dataloader:
    batch.pop('text', None)
    batch.pop('meta', None)
    outputs = model.generate(**batch)
    print(outputs)
    break


for name, mod in model.named_modules():
    print(name)
