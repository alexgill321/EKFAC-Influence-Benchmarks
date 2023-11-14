import torch

def calculate_bergman_divergance(trainloader, example_prediction, trained_model_example_prediction, criterion, output_grads):
    # let's do only for one batch, hence break
    for inputs, labels in trainloader:

        itr = 0

        for individual_inputs_in_batch, individual_label_in_batch in zip(inputs, labels):
            loss_on_untrained_examples = criterion(example_prediction[0][itr],
                                                   individual_label_in_batch.unsqueeze(dim=0))

            loss_on_trained_examples = criterion(trained_model_example_prediction[0][itr],
                                                 individual_label_in_batch.unsqueeze(dim=0))

            # grad of final prediction wrt loss (not sure):
            output_grads_colum_vector = output_grads[0][itr].t()

            # difference_in_preds_before_vs_after_training
            output_difference_vector = example_prediction[0][itr] - trained_model_example_prediction[0][itr]

            output_matmul = output_grads_colum_vector * output_difference_vector

            final_bergman_for_current_input = loss_on_untrained_examples - loss_on_trained_examples - output_matmul

            print(final_bergman_for_current_input)

            itr += 1
            break
        break

    return final_bergman_for_current_input


def pbrf_from_bergman(bergman_divergance, untrained_model_params,trained_model_params, ):
    pbrf_for_curr_example = bergman_divergance.item() - torch.sum(
        torch.square(untrained_model_params - trained_model_params))
    print(pbrf_for_curr_example)


