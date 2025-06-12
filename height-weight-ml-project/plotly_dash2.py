import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import numpy as np

# Import data
df = pd.read_csv(r'D:\ML_project\LINEAR_REGRESSION\PROJECT\height-weight-ml-project\data\height-weight.csv') 

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])


app.layout = dbc.Container([
    html.H1("Heigh-Weight Dashboard", className="text-center my-4"),
    
    # Dropdown selectors (outside tabs so they're always visible)
    dbc.Row([
        dbc.Col([
            html.Label("Select X-axis Feature:"),
            dcc.Dropdown(
                id='x-feature',
                options=[{'label': col, 'value': col} for col in df.columns],
                value='Height'
            ),
        ], width=6),
        
        dbc.Col([
            html.Label("Select Y-axis Feature:"),
            dcc.Dropdown(
                id='y-feature',
                options=[{'label': col, 'value': col} for col in df.columns],
                value='Weight'
            ),
        ], width=6),
    ]),
    
  
    dcc.Tabs([
        # Tab1 - Distribution Plots
        dcc.Tab(label='Distribution', children=[
           
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='histogram')
                ], width=12)
            ])
        ]),
        
        # Tab 2 - Scatter Plot
        dcc.Tab(label='Scatter Plot', children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='scatter-plot')
                ], width=12)
            ])
        ]),
        
        # Tab 3 - Raw Data
        dcc.Tab(label='Raw Data', children=[
            dbc.Row([
                dbc.Col([
                    html.Div(
                        id='data-table',
                        style={'height': '500px', 'overflowY': 'scroll'}
                    )
                ], width=12)
            ])
        ])
    ])
], fluid=True)



@app.callback(
    Output('histogram', 'figure'),
    Input('x-feature', 'value'),
)
# def update_histogram(feature):
#     fig = px.histogram(df, x=feature,histnorm='probability density', nbins=50, title=f'Histogram of {feature}')
#     return fig

def update_kde_plot(feature):
   
    data = df[feature].dropna().values
    kde = gaussian_kde(data)
    
   
    x_vals = np.linspace(data.min(), data.max())
    y_vals = kde.evaluate(x_vals)
    
  
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='KDE'))
    fig.update_layout(
        title=f'Distribution (KDE) of {feature}',
        xaxis_title=feature,
        yaxis_title='Density'
    )
    return fig

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-feature', 'value'), Input('y-feature', 'value')]
)
def update_scatter(x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
    fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))
    return fig

@app.callback(
    Output('data-table', 'children'),
    [Input('x-feature', 'value'), Input('y-feature', 'value')]
)
def update_table(x_col, y_col):
    
    return dash.dash_table.DataTable(
        data=df[[x_col, y_col]].head(100).to_dict('records'),
        columns=[{'name': i, 'id': i} for i in [x_col, y_col]],
        page_size=10,
        style_table={'overflowX': 'auto'}
    )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)