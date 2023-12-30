#%%
from torchvision import datasets, transforms
from torch.utils.data import Subset
import os
import ast
import plotly.graph_objects as go

train_dataset = datasets.MNIST(root='../data', train=True, download=True)
test_dataset = Subset(train_dataset, range(500))

#%%
array_list = []
with open(os.getcwd() + '/top_influences.txt', 'r') as file:
    for _ in range(2):  # Skip the first two lines
        file.readline()

    for line in file:
        line = line.strip()
        splits = line.split(': ')
        if len(splits) != 2:
            continue
        try:
            list = ast.literal_eval(splits[1])
            array_list.append(list)
        except ValueError as e:
            continue


# %%
for i, list in enumerate(array_list):
    fig = go.Figure()

# Iterate over image paths to add each image to the subplot
    for j, index in enumerate(list):
        # Open the image file
        image = train_dataset[index][0]

        # Add image to subplot
        fig.add_trace(
            go.Image(
                z=image,
                hoverinfo='none',
                x0=i,
                dx=1,
                y0=0,
                dy=1,
            )
        )

    # Update layout
    fig.update_layout(
        xaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=False,
        ),
        images=[
            dict(
                source=train_dataset[index][0],
                x=i,
                y=0,
                sizex=1,
                sizey=1,
                sizing="stretch",
                opacity=1,
                layer="below",
            )
            for i, index in enumerate(list)
        ],
    )

    # Show the plot
    fig.show()
    break
# %%
