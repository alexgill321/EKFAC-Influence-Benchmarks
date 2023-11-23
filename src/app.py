# import dash
# import dash_core_components as dcc
from dash import dcc, html, Dash, Input, Output, State
from dash_canvas import DashCanvas, utils
import numpy as np
import cv2
#import matplotlib.pyplot as plt

app = Dash(__name__)

app.layout = html.Div([
    DashCanvas(id='canvas', width=280, height=280),
    html.Div(id='status'),
    dcc.Graph(id='bar-chart'),
])


# Predict button callback
@app.callback(
    Output('status', 'children'),
    Output('bar-chart', 'figure'),
    Input('canvas', 'trigger'),
    State('canvas', 'json_data'),
    prevent_initial_call=True
)
def predict(n_clicks, canvas_data_url):
    # Replace this with the appropriate endpoint for your Python script
    prediction_url = 'http://example.com/mnist'
    
    
    # Simulate the prediction by decoding the image data (base64) and processing it
    if canvas_data_url:
        a = utils.parse_json.parse_jsonstring(canvas_data_url, (280,280))
        image = a.astype(float)
        image = cv2.resize(image, None, fx = 0.1, fy = 0.1)
        # Example: Perform some processing on the image data (replace this with your logic)
        
        prediction_result = [1] * 10  # Replace with the actual prediction result
        # Return the result
        return 'Prediction successful', {
            'data': [{'x': [str(i) for i in range(10)], 'y': prediction_result, 'type': 'bar'}],
            'layout': {'title': 'Prediction Result'}
        }
    else:
        return 'No canvas data to predict', {'data': [], 'layout': {}}

if __name__ == '__main__':
    app.run_server(debug=True)

