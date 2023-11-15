import torch
import random

import sys
sys.path.append('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/')

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.linear_nn import load_data, get_model, load_model, test, load_checkpoint

from pbrf.pbrf_helpers import calculate_bergman_divergance, pbrf_from_bergman

train_loader, val_loader, test_loader = load_data()
print("got the data.")
model, criterion, optimizer = get_model()
model1, criterion1, optimizer1 = get_model()


untrained_model, _, _, _= load_checkpoint(model, optimizer, filepath = 'models/checkpoints/checkpoint_1_linear_trained_model.pth')
trained_model = load_model(model1, filepath = 'models/linear_trained_model.pth')

print("got the model.")

def compare_state_dicts(state_dict1, state_dict2):
    for (key1, val1), (key2, val2) in zip(state_dict1.items(), state_dict2.items()):
        if not torch.equal(val1, val2):
            print(f"Difference found in key: {key1}")
            return False
    return True

state_dict1 = untrained_model.state_dict()
state_dict2 = trained_model.state_dict()

are_identical = compare_state_dicts(state_dict1, state_dict2)
print("Are models identical?", are_identical)

# print(untrained_model.fc1.weight)
# print("***")
# print(trained_model.fc1.weight)
# print(torch.eq(untrained_model.fc1.weight, trained_model.fc1.weight))
# exit()

untrained_model_params = untrained_model.fc1.weight
trained_model_params = trained_model.fc1.weight
print(untrained_model_params)
print(trained_model_params)
print(torch.eq(untrained_model_params,trained_model_params))
diff_tensor = trained_model_params - untrained_model_params
print(torch.sum(diff_tensor)) # to confirm the weights are not exactly the same

training_preds_on_untrained_model, _ = test(untrained_model, '' , criterion, get_preds_only = True, train_loader = train_loader)
training_preds_on_trained_model, output_grads = test(trained_model, '' , criterion, get_preds_only = True, train_loader = train_loader)

# one_example_bergman_cov = calculate_bergman_divergance(train_loader, training_preds_on_untrained_model,
#                                                        training_preds_on_trained_model, criterion, output_grads)
# pbrf_from_bergman(one_example_bergman_cov, untrained_model_params, trained_model_params)


full_dataset_bergman_cov = calculate_bergman_divergance(train_loader, training_preds_on_untrained_model,
                                                       training_preds_on_trained_model, criterion, output_grads)
full_dataset_pbrf = pbrf_from_bergman(full_dataset_bergman_cov, untrained_model_params, trained_model_params)

print(full_dataset_pbrf[:10])

