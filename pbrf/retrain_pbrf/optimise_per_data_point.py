import os

import torch
import random
from tqdm import tqdm


import sys
sys.path.append('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/')


from src.linear_nn import load_data, get_model, load_model, test, load_model

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from pbrf.retrain_pbrf.helper_methods import transform_and_choose_train_data_for_scores


def train_per_data_point(train_loader, pbrf_optimizer, original_optimizer, epochs):
    checkpoint_dir = './models/pbrf/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_examples_for_if_scores, test_examples_for_if_scores, random_train, random_test = \
        (transform_and_choose_train_data_for_scores())

    train_examples_for_bregman_divergence, _, random_train_for_bregman_divergence, _ = \
        (transform_and_choose_train_data_for_scores(size_for_train_set=20))

    #got 500 examples from the source, now with each of them, converge till optimisation for k/2 steps.
    # we also have 20 random examples for calculation of bregman divergence

    '''
    algorithm : 
    for example in 500 random examples, 
        for k/2 steps for each example
            get BD, loss, proximity gap
            calculate loss, take step, get new params
        once done, we have final "close to optimal params for THETA curr for this training example"
        for each test example, get score for this example (loss with new param vs loss before)
        discard params, move onto next example
    '''
    # itr_num = 0
    if_scores_for_all_random_train_examples = {}

    for test_idx in random_test:
        if_scores_for_all_random_train_examples[test_idx.item()] = {}

    # for inputs, labels in train_examples_for_if_scores:
    for itr_num, (inputs, labels) in enumerate(tqdm(train_examples_for_if_scores)):

        original_optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)

        warm_start, criterion, optimizer2 = get_model()
        pbrf_optimizer.model = load_model(warm_start, filepath='models/checkpoints/checkpoint_6_linear_trained_model.pth')

        for steps in range(30):


            optimizer2.zero_grad()
            outputs_wrt_pbrf_optimizer = pbrf_optimizer.model(inputs)
            outputs_wrt_original_optimizer = pbrf_optimizer.base_model(inputs)


            loss_with_pbrf_modified_term = pbrf_optimizer.step(outputs_wrt_pbrf_optimizer,
                                                               outputs_wrt_original_optimizer,
                                                               labels,
                                                               train_examples_for_bregman_divergence,
                                                               original_optimizer)
            loss_with_pbrf_modified_term.backward()
            optimizer2.step()

        test_ex_num = 0
        for test_input, test_label in test_examples_for_if_scores:
            optimizer2.zero_grad()

            pbrf_optimizer.base_model.eval()
            pbrf_optimizer.model.eval()

            ###get the loss with org model, then this model then get influence
            test_output_with_baseline = pbrf_optimizer.base_model(test_input)
            test_output_with_pbrf_optimised_model = pbrf_optimizer.model(test_input)

            # Compute cross-entropy loss
            try:
                loss_base = criterion(test_output_with_baseline, test_label)
                loss_pbrf_model = criterion(test_output_with_pbrf_optimised_model, test_label)
            except Exception as e:

                # Compute mean squared error
                loss_base = criterion(test_output_with_baseline, torch.nn.functional.one_hot(test_label, num_classes=10).float())
                loss_pbrf_model = criterion(test_output_with_pbrf_optimised_model, torch.nn.functional.one_hot(test_label, num_classes=10).float())

            if_score_for_this_example = loss_pbrf_model.item() - loss_base.item()


            if_scores_for_all_random_train_examples[random_test[test_ex_num].item()][itr_num] = if_score_for_this_example

            test_ex_num+=1

        # if itr_num>10:
        #     break

        # itr_num+=1

    with open(os.getcwd() + f'/pbrf/retrain_pbrf/retrained_pbrf_scores_fc1.txt', 'w') as file:
        for test_example in if_scores_for_all_random_train_examples:
            file.write(f'{test_example}: {list(if_scores_for_all_random_train_examples[test_example].values())}\n')
    file.close()
    # for key in if_scores_for_all_random_train_examples:
    #     print(key, '\t', if_scores_for_all_random_train_examples[key])
    #     print("\n")
    #




    # for epoch in range(epochs):
    #
    #     if_scores_for_all_random_train_examples = {}
    #
    #     for test_idx in random_test:
    #         if_scores_for_all_random_train_examples[test_idx.item()] = {}
    #
    #     pbrf_optimizer.model.train()
    #     total_loss = 0.0
    #     correct = 0
    #     total = 0
    #
    #     itr = 0
    #     for inputs, labels in train_loader:
    #
    #         optimizer2.zero_grad()
    #
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #
    #         outputs = pbrf_optimizer.model(inputs)
    #         # Compute cross-entropy loss
    #         try:
    #             loss = criterion(outputs, labels)
    #         except:
    #             # Compute mean squared error
    #             loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
    #
    #         # loss.backward() we will not make any weight update using the loss function/optim now,
    #         # only using PBRF
    #
    #         loss_with_pbrf = pbrf_optimizer.step(outputs, loss, criterion)
    #         loss_with_pbrf.backward()
    #         optimizer2.step()
    #
    #         if itr in random_train:
    #             # if inputs.shape == first_input_tensor.shape:
    #             #     # Ensure both tensors are of the same data type
    #             #     if inputs.dtype == first_input_tensor.dtype:
    #             #         # Compare tensors for equality
    #             #         if torch.all(torch.eq(inputs[200:210], first_input_tensor[200:210])):
    #             #             print("correct and equal")
    #             #         else:
    #             #             print(inputs[200:210])
    #             #             print(first_input_tensor[200:210])
    #             #             print("not equal")
    #             #     else:
    #             #         print("Tensors have different data types")
    #             # else:
    #             #     print("Tensors have different shapes")
    #
    #
    #             test_ex_num = 0
    #             pbrf_optimizer.base_model.eval()
    #             pbrf_optimizer.model.eval()
    #
    #             for test_input, test_label in test_examples_for_if_scores:
    #                 # print(pbrf_optimizer.model.fc1.weight)
    #                 optimizer2.zero_grad()
    #                 ###get the loss with org model, then this model then get influence
    #                 test_output_with_baseline = pbrf_optimizer.base_model(test_input)
    #                 test_output_with_pbrf_optimised_model = pbrf_optimizer.model(test_input)
    #
    #                 # Compute cross-entropy loss
    #                 try:
    #                     loss_base = criterion(test_output_with_baseline, test_label)
    #                     loss_pbrf_model = criterion(test_output_with_pbrf_optimised_model, test_label)
    #                 except Exception as e:
    #
    #                     # Compute mean squared error
    #                     loss_base = criterion(test_output_with_baseline, torch.nn.functional.one_hot(test_label, num_classes=10).float())
    #                     loss_pbrf_model = criterion(test_output_with_pbrf_optimised_model, torch.nn.functional.one_hot(test_label, num_classes=10).float())
    #
    #                 if_score_for_this_example = loss_base.item() - loss_pbrf_model.item()
    #
    #                 print(test_ex_num, itr,loss_base, loss_pbrf_model)
    #
    #                 if_scores_for_all_random_train_examples[random_test[test_ex_num].item()][itr] = if_score_for_this_example
    #
    #                 # if_scores_for_all_random_train_examples[random_test[test_ex_num].item()][itr] = if_score_for_this_example
    #
    #                 test_ex_num+=1
    #                 break
    #
    #         print("\n")
    #         pbrf_optimizer.base_model.train()
    #         pbrf_optimizer.model.train()
    #         itr+=1
    #
    #         # total_loss += loss.item()
    #         # Compute accuracy
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    #     print(if_scores_for_all_random_train_examples[8429])
    #     exit()
    #     avg_loss = total_loss / len(train_loader)
    #     accuracy = correct / total
    #
    #     # Print validation loss during training
    #     # val_loss, val_acc = validate(pbrf_optimizer.model, val_loader, criterion)
    #     # print(
    #     #     f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%")
    #
    #
    #     if epoch % 5 == 0:
    #         # Save checkpoint
    #         checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch + 1}_linear_trained_model_small.pth')
    #         torch.save(pbrf_optimizer.model.state_dict(), checkpoint_path)
    #         print(f"Checkpoint saved at {checkpoint_path}")
    #
    # # Save the trained model
    # torch.save(pbrf_optimizer.model.state_dict(), './models/linear_trained_model_small.pth')
    # print("Trained model saved.")
