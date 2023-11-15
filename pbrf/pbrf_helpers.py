import torch
from tqdm import tqdm

def calculate_bergman_divergance(trainloader, example_prediction, trained_model_example_prediction, criterion, output_grads):


    bergman_divergence_for_entire_dataset = []
    # let's do only for one batch, hence break

    for batch_num, data_batch in enumerate(tqdm(trainloader)):
    # for inputs, labels in trainloader:


        inputs, labels = data_batch
        itr = 0

        for individual_inputs_in_batch, individual_label_in_batch in zip(inputs, labels):

            loss_on_untrained_examples = criterion(example_prediction[batch_num][itr].unsqueeze(dim=0),
                                                   individual_label_in_batch.unsqueeze(dim=0))

            loss_on_trained_examples = criterion(trained_model_example_prediction[batch_num][itr].unsqueeze(dim=0),
                                                 individual_label_in_batch.unsqueeze(dim=0))

            output_grads[batch_num] = output_grads[batch_num].argmax(dim = 1).unsqueeze(dim = 1)

            example_pred_curr_with_argmax = example_prediction[batch_num][itr].argmax(dim = 0)
            trained_model_example_pred_curr_with_argmax = trained_model_example_prediction[0][itr].argmax(dim = 0)

            output_grads_colum_vector = output_grads[batch_num][itr].t()

            # difference_in_preds_before_vs_after_training
            output_difference_vector = example_pred_curr_with_argmax - trained_model_example_pred_curr_with_argmax


            output_matmul = output_grads_colum_vector * output_difference_vector

            final_bergman_for_current_input = loss_on_untrained_examples - loss_on_trained_examples - output_matmul

            bergman_divergence_for_entire_dataset.append(final_bergman_for_current_input)

            itr += 1

    return bergman_divergence_for_entire_dataset


def pbrf_from_bergman(bergman_divergance, untrained_model_params,trained_model_params, ):

    full_dataset_pbrf = []
    for each_example_divergence in bergman_divergance:
        pbrf_for_curr_example = each_example_divergence.item() - torch.sum(
            torch.square(untrained_model_params - trained_model_params))

        full_dataset_pbrf.append(pbrf_for_curr_example)

    return full_dataset_pbrf



