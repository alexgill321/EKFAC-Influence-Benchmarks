# import dash
# import dash_core_components as dcc
from dash import dcc, html, Dash, Input, Output, State
from dash_canvas import DashCanvas, utils
from dash import callback_context
import numpy as np
import cv2
#import matplotlib.pyplot as plt
from linear_nn import get_model, load_model
import torch
import os
from torchinfo import summary
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from compare_pbrf_if_performance import get_data_from_if_tensors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set the random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def load_data(store_mnist='../data'):
    # Define the transformation to flatten the images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Lambda(lambda x: x)])

    # Download MNIST dataset and create DataLoader
    train_dataset = datasets.MNIST(root=store_mnist, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=store_mnist, train=False, transform=transform, download=True)

    # Split the training dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = load_data()

def run_model(img : np.array):
    net, _, _ = get_model()
    model = load_model(net, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/linear_trained_model.pth'))
    x_data = torch.tensor(img).view(1,-1)
    
    print(x_data.shape)
    model.eval()
    with torch.no_grad():
        results = model(x_data)
    print(torch.max(results.data, 1))
    return results[0]
    
def create_subplots(image_list, titles):
    # Create a subplot with a 5x5 grid
    fig = make_subplots(rows=5, cols=5, subplot_titles=titles)

    # Add traces for each subplot
    for i, image in enumerate(image_list, start=1):
        row_num = (i - 1) // 5 + 1
        col_num = (i - 1) % 5 + 1
        subplot_trace = go.Heatmap(z=image, colorscale='Viridis', showscale=False)
        #subplot_trace.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain')
        #subplot_trace.update_xaxes(constrain='domain')
        fig.add_trace(subplot_trace, row=row_num, col=col_num)

    # Update layout
    fig.update_layout(height=800, width=800, title_text="Top 25 Training Examples")
    
    return fig

index = 0
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='compressed_image'),
        html.Button(id='left_data', value='Previous'), 
        html.Button(id='right_data', value='Next'), 
        dcc.Input(id='index_holder', value=index)
    ], style={'display': 'inline-block', 'width': '400px'}),
    
    html.Div(id='status'),
    dcc.Graph(id='histogram'),
    dcc.Graph(id='top25'),
])

@app.callback(
    [
        Output('compressed_image', 'figure'),
        Output('index_holder', 'value'),
        Output('histogram', 'figure'),
        Output('top25', 'figure')
    ],
    [
        Input('left_data', 'n_clicks'),
        Input('right_data', 'n_clicks'),
        Input('index_holder', 'value')
    ]
)
def move_index(nl, nr, index):
    index = int(index)
    trigger = callback_context.triggered[0]['prop_id'].split('.')[0]
    print(trigger) 
    if trigger == 'left_data':
        index -= 1
    elif trigger == 'right_data':
        index += 1
    

    image, val = train_dataset[index]
    #print(image)
    fig = px.imshow(image[0])

    fig1, top_indices, top_values, max_indices_associated_labels, correlation_coefficient = get_data_from_if_tensors(index, show=False)
    fig2 = create_subplots([np.flipud(train_dataset[i][0][0]) for i in top_indices[:25]], [f'Value {np.round(v)}' for v in top_values[:25]])

    return fig, int(index), fig1, fig2
 

if __name__ == '__main__':
    app.run_server(debug=True)

