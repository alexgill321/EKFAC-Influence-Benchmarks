import os

import torch
import random

import sys
sys.path.append('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/')

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from pbrf.retrain_pbrf.optimise_per_data_point import train_per_data_point
from src.linear_nn import load_data, get_model, load_model, test, load_model
from pbrf.retrain_pbrf.helper_methods import PBRFOptimizer, validate, transform_and_choose_train_data_for_scores

'''
Step 1: Get the partially trained model, note the weights, 
Step 2: Load the same model (warm start) and start a new training loop, for 50% epochs.
Weight update is done using PBRF
'''

train_loader, test_loader = load_data(validation_required=False)
print("got the data.")
model1, _, optimizer1 = get_model()
model2, criterion, optimizer2 = get_model()

base_model = load_model(model1, filepath = 'models/checkpoints/checkpoint_6_linear_trained_model.pth')
model_for_pbrf = load_model(model2, filepath = 'models/checkpoints/checkpoint_6_linear_trained_model.pth')

print("got the model.")
theta_s = base_model.fc1.weight


def train(train_loader, criterion, pbrf_optimizer, epochs):

    checkpoint_dir = './models/pbrf/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_examples_for_if_scores, test_examples_for_if_scores, random_train, random_test = (
        transform_and_choose_train_data_for_scores())
    for epoch in range(epochs):
        if_scores_for_all_random_train_examples = {}
        for test_idx in random_test:
            if_scores_for_all_random_train_examples[test_idx.item()] = {}

        pbrf_optimizer.model.train()
        correct = 0
        total = 0
        itr = 0
        for inputs, labels in train_loader:
            optimizer2.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = pbrf_optimizer.model(inputs)
            try:
                loss = criterion(outputs, labels)
            except:
                loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
            loss_with_pbrf = pbrf_optimizer.step(outputs, loss)
            loss_with_pbrf.backward()
            optimizer2.step()
            if itr in random_train:
                test_ex_num = 0
                pbrf_optimizer.base_model.eval()
                pbrf_optimizer.model.eval()
                for test_input, test_label in test_examples_for_if_scores:
                    optimizer2.zero_grad()
                    test_output_with_baseline = pbrf_optimizer.base_model(test_input)
                    test_output_with_pbrf_optimised_model = pbrf_optimizer.model(test_input)
                    try:
                        loss_base = criterion(test_output_with_baseline, test_label)
                        loss_pbrf_model = criterion(test_output_with_pbrf_optimised_model, test_label)
                    except Exception as e:
                        loss_base = criterion(test_output_with_baseline, torch.nn.functional.one_hot(test_label, num_classes=10).float())
                        loss_pbrf_model = criterion(test_output_with_pbrf_optimised_model, torch.nn.functional.one_hot(test_label, num_classes=10).float())
                    if_score_for_this_example = loss_base.item() - loss_pbrf_model.item()
                    print(test_ex_num, itr,loss_base, loss_pbrf_model)
                    if_scores_for_all_random_train_examples[random_test[test_ex_num].item()][itr] = if_score_for_this_example
                    test_ex_num+=1
                    break
            print("\n")
            pbrf_optimizer.base_model.train()
            pbrf_optimizer.model.train()
            itr+=1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(if_scores_for_all_random_train_examples[8429])
        exit()
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        if epoch % 5 == 0:
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch + 1}_linear_trained_model_small.pth')
            torch.save(pbrf_optimizer.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # Save the trained model
    torch.save(pbrf_optimizer.model.state_dict(), './models/linear_trained_model_small.pth')
    print("Trained model saved.")

pbrf_optimizer = PBRFOptimizer( base_model_weights = theta_s,
                                base_model = base_model,
                                current_model = model_for_pbrf,
                                damping= 0.001,
                                learning_rate= 10,
                                criterion=criterion)

# train(train_loader, criterion, pbrf_optimizer, 2)
train_per_data_point(train_loader, pbrf_optimizer, original_optimizer= optimizer2, epochs=2)

# training_preds_on_untrained_model, _ = test(untrained_model, '' , criterion, get_preds_only = True, train_loader = train_loader)
# training_preds_on_trained_model, output_grads = test(trained_model, '' , criterion, get_preds_only = True, train_loader = train_loader)
#
#
# full_dataset_bergman_cov = calculate_bergman_divergance(train_loader, training_preds_on_untrained_model,
#                                                        training_preds_on_trained_model, criterion, output_grads)
# full_dataset_pbrf = pbrf_from_bergman(full_dataset_bergman_cov, untrained_model_params, trained_model_params)
# a = torch.stack(full_dataset_pbrf)
# print("saving model now")
# torch.save(a, 'pbrf_tensor.pt')

