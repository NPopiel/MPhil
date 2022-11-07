import dash
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.exceptions import PreventUpdate

from dash import dcc, html, dash_table

########### Layout:
app = dash.Dash(__name__)

event_df = pd.DataFrame({"id": [0,1,2], "a": [11,21,31], "b": [41,51,61]})
PAGE_SIZE=1

app.layout = html.Div(children=[
    html.Div(id='data-storage-json', style={'display': 'none'}),
    html.Div(children=[
                dash_table.DataTable(
                        id='event-table',
                        style_data={'whiteSpace': 'normal'}, #'border': '1px solid blue'},
                        style_cell={'textAlign': 'center'},
                        #style_header={ 'border': '1px solid pink' },
                        css=[{
                            'selector': '.dash-cell div.dash-cell-value',
                            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                        }],
                        columns=[{"name": i, "id": i} for i in event_df.columns if i is not 'id'],
                        style_table={'overflowX': 'scroll'},
                        row_selectable='single',
                        selected_rows=[],
                        page_current=0,
                        page_size=PAGE_SIZE,
                        page_action='custom',
                        filter_action='custom',
                        filter_query='',
                        sort_action='custom',
                        sort_mode='multi',
                        sort_by=[]
                  ),
                  html.Div(id='event-stats', style={'width': '80%', 'color': 'black', 'font-size': '9'})],
                  style={'width': '90%', 'margin-left': '20px', 'font-size': '9', 'horizontal-align': 'middle', 'vertical-align': 'middle'}),
    html.Div(children=[html.Br()]),
    html.Button('Plot', id='show-button'),
    html.Div(id='plot-div', children=[], style={'width': '95%', 'font-size': '9', 'vertical-align': 'middle'}),
])

########### Callbacks:

'''
Callback for sorting/filtering table
'''
@app.callback(
Output('event-table', 'data'),
[Input('event-table', 'sort_by'),
 Input('event-table', 'filter_query'),
 Input('event-table', 'page_current'),
 Input('event-table', 'page_size')])
def update_event_selection(sort_by, filter_query,page_current, page_size):

    return event_df.to_dict('records')

@app.callback(
Output('data-storage-json','children'),
[Input('show-button', 'n_clicks')],
[State('event-table','selected_row_ids')
])
def prepare_data(n_clicks,selected_id):
    duration=1

    print('Selected id: ',selected_id)

    if n_clicks is None or  selected_id is None or len(selected_id)==0:
        raise PreventUpdate

    duration=int(duration)
    selected_id=selected_id[0]
    row=event_df.loc[selected_id,:]
    print(row)

    res_df = pd.DataFrame({"id": [0,1,2], "a": [11,21,31], "b": [41,51,61]})
    js=res_df.to_json(date_format='iso', orient='split')
    print('In Prep: ',len(js))
    return js

@app.callback(
Output('plot-div','children'),
[Input('data-storage-json','children')],
[State('event-table','selected_row_ids')])
def generate_plots(data_storage,selected_id):
    if data_storage is None:
        print('None!!!')
        raise PreventUpdate
    else:
        print('InDisplay -storage: '+str(len(data_storage)))
        res_df = pd.read_json(data_storage, orient='split')

    print('InDisplay ',res_df.shape)
    selected_id=selected_id[0]
    row=event_df.loc[selected_id,:]
    event_time=pd.to_datetime(row['Start'],errors='ignore')
    event_type=row['Event']+': '+row['Cause']
    event_pid=''

    # columns sorted in reverse alphabetical
    flist=sorted(np.unique([c.split('__')[1] for c in res_df.columns]))[::-1]
    print('To plot: ',res_df.shape)
    # generate plots for each type of sensor:
    fig_list=[]
    for feature in flist:
        col_list = [c for c in res_df.columns if not c.startswith('_') and c.endswith('_'+feature)]
        temp_df = res_df[col_list]
        # plot results
        print('Preparing figure '+feature)
        fig=temp_df.iplot(kind='scatter',mode='markers',size=3, title="Plot {}: {} {} {}".format(feature,event_time,event_type,event_pid), asFigure=True)
        #fig_list.append(fig)
        fig_list.append((html.Div(children=[dcc.Graph(id=feature+'-scatter',figure=fig)])))
    print('Figure done')
    return fig_list


########### Run the app:

if __name__ == '__main__':

    app.run_server(debug=True)
