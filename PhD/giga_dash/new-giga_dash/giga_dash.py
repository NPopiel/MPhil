from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
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



app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE],
           suppress_callback_exceptions=True)



def set_iso_layout(
    iso: str,
    full_name: str,
    description: str,
    month: str,
    mae: float,
    model_description: str,
    peak_data: dict,
    load_duration_curves: dict
):
    layout = html.Div([
        html.Div(id=f'{iso}-content'),
        html.Br(),
        html.Br(),
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-data-upload'),
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(html.H1(full_name), width=9),
            dbc.Col(width=2),
        ], justify='center'),
        dbc.Row([
            dbc.Col(
            html.Div(children=description), width=9),
            dbc.Col(width=2)
        ], justify='center'),
        html.Br(),
        dbc.Row([
            dbc.Col(
                html.H3('Model Performance'), width=9
            ),
            dbc.Col(width=2),
        ], justify='center'),
        dbc.Row([
            dbc.Col(
                html.Div(
                    children=f'Mean Absolute Error (MAE) for {month}, 2021: {mae}'
                ), width=9
            ),
            dbc.Col(width=2),
        ], justify='center'),
        html.Br(),
        dbc.Row([
            dbc.Col(
                    dcc.Dropdown(
                        id=f'{iso}-dropdown',
                        options=[
                            {'label': 'Actual', 'value': 'Actual'},
                            {'label': 'Predicted', 'value': 'Predicted'}
                        ],
                        value=['Actual', 'Predicted'],
                        multi=True,
                    ), width=6
            ),
            dbc.Col(width=5),
        ], justify='center'),
        dcc.Graph(id=f'{iso}-graph'),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col(html.H3('Training Data'), width=9),
            dbc.Col(width=2)
        ], justify='center'),
        dbc.Row([
            dbc.Col(
                    html.Div(children=model_description), width=9
            ),
            dbc.Col(width=2)
        ], justify='center'),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        dcc.Graph(
                            figure=plot_histogram(iso=iso, peak_data=peak_data)
                        ),
                    ]), width=4),
                dbc.Col(
                    html.Div([
                        dcc.Graph(
                            figure=plot_load_duration(
                                iso=iso,
                                load_duration_curves=load_duration_curves
                            ),
                        ),]), width=4),
                dbc.Col(
                    html.Div([
                        dcc.Dropdown(
                            id=f'{iso}-scatter-dropdown',
                            options=[
                                {'label': 'Day of Week', 'value': 'weekday'},
                                {'label': 'Season', 'value': 'season'}
                                ],
                            value='season',
                            multi=False,
                        ),
                        dcc.Graph(id=f'{iso}-scatter')
                    ]
                ), width=4),
            ]
        ),
    ])
    return layout


def plot_load_curve(value, iso: str, load: dict, predictions: dict):
    iso = iso.upper()
    fig = go.Figure()
    if 'Actual' in value:
        fig.add_trace(go.Scatter(
            x=load[iso].index,
            y=load[iso].values,
            name='Actual Load',
            line=dict(color='maroon', width=3)))
    if 'Predicted' in value:
        fig.add_trace(go.Scatter(
            x=predictions[iso].index,
            y=predictions[iso].values,
            name = 'Forecasted Load',
            line=dict(color='darkturquoise', width=3, dash='dash')))
    return fig.update_layout(
        title="System Load: Actual vs. Predicted",
        xaxis_title="Date",
        yaxis_title="Load (MW)",
        template=TEMPLATE
    )


def plot_histogram(iso: str, peak_data: dict):
    iso = iso.upper()
    return px.histogram(
        peak_data[iso],
        x=peak_data[iso]['load_MW'],
        nbins=75,
        marginal="rug",
        title=f"Distribution of {iso} Daily Peaks",
        color_discrete_sequence=['darkturquoise']
    ).update_layout(
        template=TEMPLATE,
        xaxis_title='Peak Load (MW)',
        yaxis_title='Number of Days'
    )


def plot_scatter(value, iso: str, peak_data: dict):
    fig = px.scatter(
        peak_data[iso.upper()].dropna(),
        x="load_MW",
        y="temperature",
        color=value
    )
    return fig.update_layout(
        template=TEMPLATE, title='Peak Load vs. Temperature'
    )

def plot_load_duration(iso: str, load_duration_curves: dict):
    return go.Figure().add_trace(
        go.Scatter(
            x=load_duration_curves[iso.upper()].reset_index().index,
            y=load_duration_curves[iso.upper()].values,
            mode = 'lines',
            fill='tozeroy',
            line=dict(color='maroon', width=3)
        )).update_layout(
            title="Peak Load Sorted by Day (Highest to Lowest)",
            xaxis_title="Number of Days",
            yaxis_title="Load (MW)",
            template=TEMPLATE)

