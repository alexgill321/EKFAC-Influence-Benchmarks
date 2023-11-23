# import dash
# import dash_core_components as dcc
from dash import dcc, html, Dash, Input, Output, State
from dash_canvas import DashCanvas, utils
import numpy as np
import cv2
#import matplotlib.pyplot as plt
from linear_nn import get_model, load_model
import torch
import os
from torchinfo import summary
import plotly.express as px



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
    

h = 1
w = 10
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        DashCanvas(id='canvas', width=200, height=200, lineWidth=10)
    ]),#, style={'width': '49%', 'display': 'inline-block', 'height': '400px'}),
    dcc.Graph(id='compressed_image'),
    html.Div(id='status'),
    dcc.Graph(id='bar-chart'),
])


# Predict button callback
@app.callback(
    Output('status', 'children'),
    Output('bar-chart', 'figure'),
    Output('compressed_image', 'figure'),
    Input('canvas', 'trigger'),
    State('canvas', 'json_data'),
    prevent_initial_call=True
)
def predict(n_clicks, canvas_data_url):
    # Replace this with the appropriate endpoint for your Python script
    prediction_url = 'http://example.com/mnist'
    
    
    # Simulate the prediction by decoding the image data (base64) and processing it
    if canvas_data_url:
        a = utils.parse_json.parse_jsonstring(canvas_data_url, (200,200))
        image = a.astype(np.float32)
        image = cv2.resize(image, (28,28))
        image = image.astype(np.float32)
        # Example: Perform some processing on the image data (replace this with your logic)
        fig = px.imshow(image)
        # prediction_result = [1] * 10  # Replace with the actual prediction result
        prediction_result = run_model(image).detach().numpy()
        print(prediction_result)
        # Return the result
        return 'Prediction successful', {
            'data': [{'x': [str(i) for i in range(10)], 'y': prediction_result, 'type': 'bar'}],
            'layout': {'title': 'Prediction Result'}
        }, fig
    else:
        return 'No canvas data to predict', {'data': [], 'layout': {}}

if __name__ == '__main__':
    app.run_server(debug=True)

