import numpy
import torch
import matplotlib.pyplot as plt


if_tensor = torch.load('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/src/Linear(in_features=784, out_features=256, bias=True)_influences_tensor.pt')

influence_to_dataset_mapping = torch.load('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/src/l1_example_label_to_dataset_label_mapping.pt')
influence_to_dataset_mapping_new = torch.load('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/src/l1_example_label_to_dataset_label_mapping_new_conv.pt')

pbrf_tensors = torch.load('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/pbrf_tensor.pt')

print(if_tensor)



def get_plot_from_if_tensors(influence_dataset_mapping_dict):

    for itr, example in enumerate(if_tensor):

        top_indices = torch.topk(example.flatten(), 100).indices
        original_label_and_example_labels = influence_to_dataset_mapping[itr]
        original_label = list(original_label_and_example_labels.keys())[0]
        if original_label != 2:
            continue
        print(top_indices)
        print(original_label)
        print(len(original_label_and_example_labels[original_label]))
        list_for_comparison = []
        for i in top_indices.tolist():
            try:
                list_for_comparison.append(original_label_and_example_labels[original_label][i])
            except:
                pass
        print(len(list_for_comparison))
        max_indices_associated_labels = list_for_comparison
            # [original_label_and_example_labels[original_label][i] for i in top_indices.tolist()]
        print(max_indices_associated_labels)

        plt.hist(max_indices_associated_labels, bins=range(min(max_indices_associated_labels), max(max_indices_associated_labels) + 2), align='left', rwidth=0.8, color='blue', edgecolor='black')

        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.title('plot when input for IF is {}'.format(original_label))
        plt.show()

        top_result_if_tensor_score = torch.max(if_tensor, dim = 1).values

        # Stack the tensors along a new dimension to create a 2D tensor
        stacked_tensors = torch.stack([pbrf_tensors, top_result_if_tensor_score], dim=0)

        # Compute the correlation coefficient matrix
        correlation_matrix = torch.corrcoef(stacked_tensors)

        # The correlation coefficient is in the (0, 1) position of the matrix
        correlation_coefficient = correlation_matrix[0, 1]

        print("Correlation Coefficient:", correlation_coefficient.item())
        exit()


print()


get_plot_from_if_tensors(influence_to_dataset_mapping)
