# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import data_layout as l
import dash
import os
import base64
import datetime
import io
import numpy as np
import gigaanalysis as ga
from dash.exceptions import PreventUpdate

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table

import pandas as pd
from random_functions import *


app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE],
           suppress_callback_exceptions=True)

# bare bones... Needs to let you click what files you want to open
# ideally a tab / section for multiple samples
# plot all raw data with option to switch to derivative
# plot background subtracted torque and let you change poly, window, min and max field
# plot fft
# plot averages
# all the while letting you remove files from the analysis
# plot LK

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# app.layout = html.Div(children=[
#     html.H1(children='GigaDash'),
#
#     html.Div(children='''
#         A web-based application for processing quantum oscillation data
#     '''),
#
#     dcc.Graph(
#         id='example-graph',
#         figure=fig
#     )
# ],
# )

from collections import defaultdict
import pandas as pd

class keydefaultdict(defaultdict):
    """Subclasss defaultdict such that the default_factory method called for missing
    keys takes a single parameter https://stackoverflow.com/a/2912455/77533
    """
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


FILE_MAP = {
    'name1': 'file1.csv',
    'name2': 'file2.csv',
    'name3': 'file3.csv',
    #....
}

DATA_FRAMES = keydefaultdict(lambda name: pd.read_csv(FILE_MAP[name]))



"""Homepage"""

test_png = './images/beware-giga.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    ])

index_page = html.Div([
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col(html.H1(children="Welcome to GigaDash a GigaAnalysis GUI"), width=5),
            dbc.Col(width=5),
        ], justify='center'),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H4(
                        children=(
                            "A software package for analysing quantum oscillations.",
                            "I hope you find those juicy wiggles!"
                            )),
                    html.Div(l.BUTTON_LAYOUT)]), width=7
            ),
            dbc.Col(width=3),
        ], justify="center"),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),


        html.Div([
    html.Img(src='data:image/png;base64,{}'.format(test_base64)),
    ])
])

'''Data I/O'''


dset = {}

meta_df = pd.DataFrame(
        columns=['Min Field (T)', 'Max Field (T)', 'Num Points', 'Direction', 'Temp (K)', 'Angle', 'Sweep'])

data_layout = l.set_data_layout(
    iso='data',
    full_name='DataHub',
    description='Load data into GigaDash for analysis!',
)



def parse_contents2(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:

            # Assume that the user uploaded a CSV file
            # In the future update this to allow someone to specify which type of file,
            # delimeter, X, Y, where to open from. Ideally have a function for each lab.

        print(filename)

        file =  io.StringIO(decoded.decode('utf-8'))
        field, volt = pd.read_csv(
                file, sep=',').values.T
        temp_of_run = filename.split('_')[0]
        angle = filename.split('_')[1]
        sweep = filename.split('_')[2]

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return {'field_'+sweep: field, 'torque_'+sweep: volt, 'temp_'+sweep : temp_of_run,'angle_'+sweep: angle,sweep:sweep}
                                                                                                                   # need to return dset and meta_df





# add a click to the appropriate store.
@app.callback(Output('dict-memory', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents2(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children




\
# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/data':
        return data_layout
    else:
        return index_page



if __name__ == '__main__':
    app.run_server(debug=True)