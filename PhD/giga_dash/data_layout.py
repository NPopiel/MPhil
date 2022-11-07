from dash import dcc, html
import dash_bootstrap_components as dbc


BUTTON_LAYOUT = [
    dcc.Link(
        html.Button('HOME', id='home-button', className="mr-1"),
        href='/'),
    dcc.Link(
        html.Button('DATA I/O', id='data-button', className="mr-1"),
        href='/data'),
    dcc.Link(
        html.Button('Raw Data', id='raw-button', className="mr-1"),
        href='/raw'),
    dcc.Link(
        html.Button('Subtractions', id='sub-button', className="mr-1"),
        href='/sub'),
    dcc.Link(
        html.Button('averages', id='avg-button', className="mr-1"),
        href='/sub'),
    dcc.Link(
        html.Button('LK', id='LK-button', className="mr-1"),
        href='/lk')
]

from dash import Dash, dcc, html

external_stylesheets = [dbc.themes.SLATE]

app = Dash(__name__, external_stylesheets=external_stylesheets)



def set_data_layout(
    iso: str,
    full_name: str,
    description: str,
):
    layout = html.Div([
        html.Div(id=f'{iso}-content'),
        html.Br(),
        dbc.Row([
            dbc.Col(
                html.Div(BUTTON_LAYOUT), width=4),
            dbc.Col(width=7),
        ], justify='center'),
        html.Br(),
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
        html.Br(),
        html.Br(),
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

        dcc.Store(id='dict-memory'),
        html.Button('DISPLAY TABLE', id='show-table-button', n_clicks=0),
    ])
    return layout

