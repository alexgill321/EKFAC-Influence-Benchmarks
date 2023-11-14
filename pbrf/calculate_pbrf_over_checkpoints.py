import torch
import random

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.linear_nn import load_data, get_model, load_model, test
from pbrf.pbrf_helpers import calculate_bergman_divergance, pbrf_from_bergman

train_loader, val_loader, test_loader = load_data()
print("got the data.")
model, criterion, mse_criterion, optimizer = get_model()

untrained_model = load_data(model, filepath = 'models/checkpoints/checkpoint_1_linear_trained_model.pth')
trained_model = load_model(model, filepath = 'models/linear_trained_model.pth')
print("got the model.")

untrained_model_params = untrained_model.fc1.weight.grad
untrained_model_params = trained_model.fc1.weight.grad

training_preds_on_untrained_model, _ = test(untrained_model, '' , criterion, mse_criterion, get_preds_only = True, train_loader = train_loader)
training_preds_on_trained_model, output_grads = test(trained_model, '' , criterion, mse_criterion, get_preds_only = True, train_loader = train_loader)

print("got the preds")
one_example_bergman_cov = calculate_bergman_divergance(train_loader, training_preds_on_untrained_model,
                                                       training_preds_on_trained_model, criterion, output_grads)
pbrf_from_bergman(one_example_bergman_cov)




