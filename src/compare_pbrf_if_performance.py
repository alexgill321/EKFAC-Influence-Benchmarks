import numpy
import torch
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go


if_tensor = torch.load('Linear(in_features=784, out_features=256, bias=True)_influences_tensor.pt')

influence_to_dataset_mapping = torch.load('l1_example_label_to_dataset_label_mapping.pt')
influence_to_dataset_mapping_new = torch.load('l1_example_label_to_dataset_label_mapping_new_conv.pt')

pbrf_tensors = torch.load('../pbrf_tensor.pt')

print(if_tensor)

def get_data_from_if_tensors(index, show=True):
    example = if_tensor[index]
    top = torch.topk(example.flatten(), 100)
    top_indices = top.indices
    top_values = top.values

    original_label_and_example_labels = influence_to_dataset_mapping[index]
    original_label = list(original_label_and_example_labels.keys())[0]
    
    list_for_comparison = []
    for i in top_indices.tolist():
        try:
            list_for_comparison.append(original_label_and_example_labels[original_label][i])
        except:
            pass
    
    max_indices_associated_labels = list_for_comparison

    # Assuming max_indices_associated_labels is your data
    data = max_indices_associated_labels

    # Create a histogram trace
    histogram_trace = go.Histogram(
        x=data,
        xbins=dict(start=min(data), end=max(data) + 1, size=1),  # Adjust the bin size as needed
        marker_color='blue',
        marker_line=dict(color='black', width=1)
    )

    # Create layout
    layout = go.Layout(
        title='Plot when input for IF is {}'.format(original_label),
        xaxis=dict(title='Number'),
        yaxis=dict(title='Frequency'),
    )

    # Create figure
    fig = go.Figure(data=[histogram_trace], layout=layout)

    if show:
        # Show the figure
        fig.show()

    top_result_if_tensor_score = torch.max(if_tensor, dim = 1).values

    # Stack the tensors along a new dimension to create a 2D tensor
    stacked_tensors = torch.stack([pbrf_tensors, top_result_if_tensor_score], dim=0)

    # Compute the correlation coefficient matrix
    correlation_matrix = torch.corrcoef(stacked_tensors)

    # The correlation coefficient is in the (0, 1) position of the matrix
    correlation_coefficient = correlation_matrix[0, 1]

    print("Correlation Coefficient:", correlation_coefficient.item())
    return fig, top_indices, top_values, max_indices_associated_labels, correlation_coefficient.item()


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



if __name__ == '__main__':
    print()


    get_plot_from_if_tensors(influence_to_dataset_mapping)
