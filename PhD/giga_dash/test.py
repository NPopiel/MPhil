import plotly.graph_objects as go
import numpy as np

from dash_extensions.enrich import DashProxy, html, dcc, Input, Output, ServersideOutput, ServersideOutputTransform

app = DashProxy(__name__, transforms=[ServersideOutputTransform()])

T0 = 1E-12  # duration of input
N = 8192    # number of points
dt = 750 * T0 / N
T = np.arange(-N / 2, N / 2) * dt
m = 1
C = 0


def envelopef(T, T0, C, m):
    U = (np.exp(-((1 + 1j * C) / 2) * ((T / T0) ** (2 * m)))).astype(complex)
    UI = np.absolute(U) ** 2
    return U, UI


z = np.arange(-10, 10)
U, UI = envelopef(T, T0, C, m)
scatter1 = go.Scatter(x=T / T0, y=UI)
figure1 = go.Figure(data=[scatter1]).update_layout()
env_graph = dcc.Graph(id='envelopesss',
                      animate=True,
                      figure=figure1.update_layout(width=600, height=600,
                                                   xaxis=dict(range=[-8, 8])))

M_slider = dcc.Slider(
    id='m_slider',
    min=1,
    max=10,
    step=1,
    value=m,
    marks={
        1: {'label': '1'},
        10: {'label': '10'}},
)

app.layout = html.Div([
    M_slider,
    dcc.Store(id='session', storage_type='local'),
    dcc.Loading(id="loading1", children=[html.Div([env_graph])], type="circle", ),
])


@app.callback(
    Output("loading1", "children"),
    ServersideOutput("session", "data"),
    [Input("m_slider", "value")])
def update_bar_chart(mn):
    U, UI = envelopef(T, T0, C, mn)
    phase = np.angle(U)
    scatter1 = go.Scatter(x=T / T0, y=UI)
    figure1 = go.Figure(data=[scatter1]).update_layout(width=600, height=600,
                                                       xaxis=dict(range=[-8, 8]))
    data = {'u': U, 'ui': UI, 'up': phase}
    env_graph = dcc.Graph(figure=figure1)
    return env_graph, data


app.run_server(port=7777)