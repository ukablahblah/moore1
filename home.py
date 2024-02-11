import sys
sys.path.append('classes')

from node import Node
from newNode import newNode
# from dataloader import dataloader

from load_model import load_model
from collections import defaultdict

import dash
import dash_cytoscape as cyto
from dash import html, dcc
from dash import html, Input, Output, State, dcc
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from dash.exceptions import PreventUpdate

data = load_model()
graph = data[0]



# Create nodes for all nodes in the graph
node_elements = [
    {'data': {'id': str(node), 'label': str(node)}} for node in graph.keys()
]


# Create edges for the graph
edge_elements = [
    {'data': {'source': str(node), 'target': str(neighbor)}} for node, neighbors in graph.items() for neighbor in neighbors
]



# node_neighbour = lambda a: data[a], list(data[a])
app = dash.Dash(__name__)

app.title = 'Robotic Registers'
app._favicon = 'logo.jpg'

colors = {
    'background': '#1e2129',
    'text': '#7FDBFF'
}

# Define the app layout
app.layout = html.Div(style={'backgroundColor': '#1e2129'}, children=[
    # Replace 'red' with the actual color value you want to use
    html.Div(
        [html.Div([html.Img(src='./assets/logo.jpg', height='75px', width='75px', style={'margin-left':'8vw', 'margin-right':'19vw', 'border-radius':'10px'}),html.H1('Robotic Registers',
                 style={'textAlign': 'center', 'color': 'aliceblue', 'margin-bottom': '10px'})], className='header'),
         html.Div(
             html.Button("Generate New Graph", id="btn-generate-graph"), className='btn-box'),
         # Cytoscape graph component
         html.Div(cyto.Cytoscape(
             id='cytoscape-graph',
             elements=[],
             layout={'name': 'grid'},  # Change 'spring' to a supported layout like 'grid'
             style={'width': '100%', 'height': '70vh', 'margin': '0px'},
             stylesheet=[{
                 'selector': 'edge',
                 'style': {"curve-style": "bezier",
                           "target-arrow-shape": "triangle",
                           'line-color': '#b0b1b5',
                           'target-arrow-color': '#b0b1b5'}
             },
                 {
                     'selector': 'node',
                     'style': {
                         'content': 'data(label)',
                         'text-halign': 'center',
                         'text-valign': 'center',
                         'width': '120px',
                         'height': '30px',
                         'background-color': 'aliceblue',
                         'border-style': 'solid',
                         'outline': 'black'

                     }
                 }]
         ), className='box'),
         html.Div(
             html.Button("Download Data", id="btn-download-txt"), className='btn-box')],
        className='box outer'),
    dcc.Download(id="download-text"),
])


@app.callback(
    Output("cytoscape-graph", "elements"),
    Input("btn-generate-graph", "n_clicks"),
    prevent_initial_call=True,
)
def update_graph(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    # Choose the next index, e.g., increment by 50
    next_index = (n_clicks - 1) * 1

    # Check if the chosen index is within the data range
    if next_index < len(data):
        # # Clear existing nodes
        # nodes_to_newNode.clear()
        # newNode_to_node.clear()

        # Update the graph with the new data
        graph = data[next_index]

        # Create nodes for all nodes in the graph
        node_elements = [
            {'data': {'id': str(node), 'label': str(node)}} for node in graph.keys()
        ]

        # Create edges for the graph
        edge_elements = [
            {'data': {'source': str(node), 'target': str(neighbor)}} for node, neighbors in graph.items() for neighbor in neighbors
        ]

        # Combine node and edge elements
        elements = node_elements + edge_elements

        return elements
    else:
        raise PreventUpdate

@app.callback(
    Output("download-text", "data"),
    Input("btn-download-txt", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    # Convert the data to a string for download
    content_str = str(data)
    return dict(content=content_str, filename="graph_dataset_5-25.pkl")

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)