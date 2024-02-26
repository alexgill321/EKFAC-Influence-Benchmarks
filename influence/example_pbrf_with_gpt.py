import os
import sys
import torch
from base import BaseInfluenceObjective
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

sys.path.append('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/')

from modules import PBRFInfluenceModule
from src.decoder_transformer import get_model, prepare_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HuggingFaceWrapperObjective(BaseInfluenceObjective):

    def train_outputs(self, model, batch):
        input_ids = torch.stack(batch, dim=0)
        return model.forward(input_ids.permute(1, 0).to(DEVICE))

    def train_loss_on_outputs(self, outputs, batch):

        input_ids = torch.stack(batch, dim=0)
        input_ids = input_ids.permute(1, 0).to(DEVICE)
        labels_gold = input_ids.clone()
        print("label gols i s{}".format(labels_gold.shape))
        # labels_gold[:, -1] = labels_gold[:, 1].clone()
        labels_gold = labels_gold[:, 1].clone()
        # [:, 1]

        print("shap of labels gols i s{}".format(labels_gold.shape))

        logits = outputs

        # .swapaxes(1, 2)
        # [:, :, :-1]
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        return loss_fn(logits, labels_gold)

    def test_loss(self, model, batch):

        outputs = model.generate(input_ids=torch.stack(batch, dim=0).permute(1, 0),
                                    output_scores=True,
                                    max_length=50,
                                    return_dict_in_generate=True)



        all_log_probs = torch.log_softmax(outputs, dim = 2)
        summed_probs = torch.sum(torch.max(all_log_probs, dim=2), dim=2)
        return summed_probs


def main():
    transformer_model, _ = get_model()
    train_dataloader, test_dataloader = prepare_dataset(dataset_name='roneneldan/TinyStories-1M', tokenizer_name='roneneldan/TinyStories-1M')

    torch.manual_seed(42)


    print(transformer_model)
    params = transformer_model.model.state_dict()

    for name, param in transformer_model.model.named_parameters():
        print(f"Parameter name: {name}, Parameter shape: {param.shape}")

    module = PBRFInfluenceModule(
        model=transformer_model,
        objective=HuggingFaceWrapperObjective(),
        layers=['model.transformer.h.7.mlp.c_proj'],
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        device=DEVICE,
        damp=1e-4,
        check_eigvals=True,
        gnh = True
    )

    train_idxs = list(range(0, 1000))
    test_idxs = list(range(0, 100))

    influences = module.influences(train_idxs, test_idxs)
    #
    # for layer in influences:
    #     with open(os.getcwd() + f'/results/pbrf_influences_{layer}.txt', 'w') as f:
    #         for i, influence in enumerate(influences[layer]):
    #             f.write(f'{i}: {influence.tolist()}\n')
    #     f.close()
    # k = 10
    # with open(os.getcwd() + '/results/pbrf_top_influences.txt', 'w') as file:
    #     for layer in influences:
    #         file.write(f'{layer}\n')
    #         for i, influence in enumerate(influences[layer]):
    #             top = torch.topk(influence, k=k).indices
    #             file.write(f'Sample {i}  Top {k} Influence Indexes: {[val for val in top.tolist()]}\n')
    #

if __name__ == '__main__':
    main()
