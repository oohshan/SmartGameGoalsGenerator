import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime as dat
import dash_helpers as dh

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Read in season data from CSV
data = pd.read_csv('./team_boxscores_v3.csv')

# The regression model
reg, prediction = dh.learn(data)

# These are the stat choices available to the user for the graphing portion of the application
stat_names = ['Assists','Assist to Turnover Ratio', 'Blocked Attempts', 'Blocks', 'Defensive Rebounds', 'Fast Break Points', 'Field Goal Attempts',
              'Field Goal Percent', 'Field Goals Made', 'Free Throw Attempts', 'Free Throw Percentage', 'Free Throws Made',
              'Offensive Rebounds', 'Personal Fouls', 'Points', 'Points Against', 'Points in the Paint', 'Points off Turnovers',
              'Rebounds', 'Steals', 'Second Chance Points', 'Team Rebounds', 'Three Point Attempts', 'Three Point Percentage',
              'Three Pointers Made', 'Turnovers', 'Two Point Attempts', 'Two Point Percentage', 'Two Pointers Made'
             ]

stat_choices = [i for i in range(28)]

# Classifies the stats if they should be displayed as a decimal number or a whole number on the sliders
float_stats = ["efg","orb_pct","ftr", "tov_pct"]
int_stats = ["assists", "blocks","defensive_rebounds", "fast_break_pts", "points_in_paint","points_off_turnovers","rebounds","steals", "turnovers","opponent_drb"]

# Dictionary of stat column names with the nicer looking alternative to be displayed on sliders
nice_names = { "assists" : "Assists", "blocks" : "Blocks", "opponent_drb" : "Opponent Defensive Rebounds",
              "fast_break_pts" : "Fast Break Points", "points_in_paint" : "Points in Paint",
              "points_off_turnovers" : "Points Off Turnovers", "rebounds" : "Rebounds", "steals" : "Steals",
              "turnovers" : "Turnovers", "tov_pct" : "Turnover Percentage", "orb_pct" : "Offensive Rebound Percentage",
              "efg": "Effective Field Goal Percentage", "ftr" : "Free Throw Rating" }

app.layout = html.Div(children=[
    html.Div(html.Img(src=app.get_asset_url('test.png'), style={'height':'10%', 'width':'10%'}), style={'display': 'inline-block'}),
    html.H1(children='NCAA Smart Game Goal Generator', style={ 'textAlign': 'center'}),

    html.Label(children='Select Team, Stats, Opponent:',
            style={'padding-right': '10px', 'padding-left': '10px',
                   'backgroundColor': '#dcdcdc', 'padding-top': '10px',
                   'borderTop': 'thin lightgrey solid'
                   }),

    html.Div([
        html.Div([
            # Allows user to select team
            dcc.Dropdown(
                id='team_select',
                options=[{'label': data[(data.team_id == team)]['market'].iloc[0], 'value': team} for team in np.unique(data.team_id)],
                value='2267a1f4-68f6-418b-aaf6-2aa0c4b291f1')
            ], style={'width': '33%', 'float': 'left', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                # Allows user to select opponent
                id='opponent_select',
                options=[{'label': data[(data.team_id == team)]['market'].iloc[0], 'value': team} for team in np.unique(data.team_id)],
                value='b795ddbc-baab-4499-8803-52e8608520ab')
            ], style={'width': '33%', 'float': 'right', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                # Allows user to select stats to graph
                id='stat_select',
                options=[{'label': stat_names[st], 'value': st} for st in range(len(stat_names))],
                value = [0,1,2,3,4,5],
                multi=True)], style={'width': '34%', 'float': 'right', 'display': 'inline-block'})
        ], style={
        'backgroundColor': '#dcdcdc',
        'padding-bottom': '15px',
        'padding-top': '5px',
    }),

    #Sample bar plot. This sample plot is not displayed with the data below. It starts off graphing the data from the two preselected teams
    html.Div([
        html.Div([
            dcc.Graph(
                id='stat_bar_plot',
                figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'Kentucky'},
                        {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'Louisville'},
                    ],
                    'layout': {
                        'title': 'Team Stats Visualization'
                    }
                }
            )
        ], style={'width': '100%', 'float': 'left', 'display': 'inline-block'} ),
    ]),

    html.Div([
        html.H1(children='Smart Goal Selector '),
        html.H2(children='Projected Win Percentage: ', id='win-percentage'),
    ], 
    
    style={'textAlign': 'center', 'padding-right': '10px',
            'padding-left': '10px','backgroundColor': '#dcdcdc',
            'padding-top': '10px', 'borderTop': 'thin lightgrey solid' }
    ),

    # Sliders
    html.Div([
        html.Div([

            html.Div(id='slider-output-container1'),

            dcc.Slider(
                id='slider-1',
                min=0,
                max=100,
                step=1,
                value=10,
            ),
    ], style={'width': '50%', 'float': 'left', 'display': 'inline-block'} ),
        html.Div([

            html.Div(id='slider-output-container2'),

            dcc.Slider(
                id='slider-2',
                min=0,
                max=100,
                step=1,
                value=10,
            ),
            
    ], style={'width': '50%', 'float': 'left', 'display': 'inline-block'} ),
        html.Div([

            html.Div(id='slider-output-container3'),

            dcc.Slider(
                id='slider-3',
                min=0,
                max=100,
                step=1,
                value=10,
            ),
            
    ], style={'width': '50%', 'float': 'left', 'display': 'inline-block'} ),
        html.Div([

            html.Div(id='slider-output-container4'),

            dcc.Slider(
                id='slider-4',
                min=0,
                max=100,
                step=1,
                value=10,
            ),
            
    ], style={'width': '50%', 'float': 'left', 'display': 'inline-block'} ),
    ]),

])

@app.callback(
    dash.dependencies.Output(component_id='stat_bar_plot', component_property='figure'),
    [dash.dependencies.Input(component_id='team_select', component_property='value'),
    dash.dependencies.Input(component_id='opponent_select', component_property='value'),
    dash.dependencies.Input(component_id='stat_select', component_property='value')]
)
def update_graph(team, opponent, stats):
    ''' Updates the bar graph when a new team, opponent, or stat is selected '''
    return dh.generate_bar_chart(team, opponent, stats, stat_names, data)

@app.callback(
    dash.dependencies.Output(component_id= 'slider-output-container1', component_property='children'),
    dash.dependencies.Output(component_id= 'slider-output-container2', component_property='children'),
    dash.dependencies.Output(component_id= 'slider-output-container3', component_property='children'),
    dash.dependencies.Output(component_id= 'slider-output-container4', component_property='children'),
    [dash.dependencies.Input(component_id='slider-1', component_property= 'value')],
    [dash.dependencies.Input(component_id='slider-2', component_property= 'value')],
    [dash.dependencies.Input(component_id='slider-3', component_property= 'value')],
    [dash.dependencies.Input(component_id='slider-4', component_property= 'value')],
    [dash.dependencies.Input(component_id='team_select', component_property='value')],
)

def update_output(stat1,stat2,stat3,stat4,team):
    ''' Updates the text displayed above the sliders '''
    output1 = nice_names[dh.overallFeatures(dh.getAllTeamMatchRecords(team, data))[0]] + ': {}'.format(stat1)
    output2 = nice_names[dh.overallFeatures(dh.getAllTeamMatchRecords(team, data))[1]] + ': {}'.format(stat2)
    output3 = nice_names[dh.overallFeatures(dh.getAllTeamMatchRecords(team, data))[2]] + ': {}'.format(stat3)
    output4 = nice_names[dh.overallFeatures(dh.getAllTeamMatchRecords(team, data))[3]] + ': {}'.format(stat4)
    return output1, output2, output3, output4

@app.callback(
    dash.dependencies.Output(component_id= 'win-percentage', component_property='children'),
    [dash.dependencies.Input(component_id='team_select', component_property='value')],
    [dash.dependencies.Input(component_id='slider-1', component_property= 'value')],
    [dash.dependencies.Input(component_id='slider-2', component_property= 'value')],
    [dash.dependencies.Input(component_id='slider-3', component_property= 'value')],
    [dash.dependencies.Input(component_id='slider-4', component_property= 'value')],
)

def update_win_percentage_with_stat_val(team, stat1, stat2, stat3, stat4):
    ''' Calculates the win percentage given the user selected stat values '''
    return dh.calculate_win_percentage(team, stat1, stat2, stat3, stat4, reg, data)

@app.callback(
    dash.dependencies.Output(component_id='slider-1', component_property= 'value'),
    dash.dependencies.Output(component_id='slider-2', component_property= 'value'),
    dash.dependencies.Output(component_id='slider-3', component_property= 'value'),
    dash.dependencies.Output(component_id='slider-4', component_property= 'value'),
    [dash.dependencies.Input(component_id='team_select', component_property='value')],
)

def set_default_values(team):
    ''' Sets default values of the sliders to be the projected values from the model '''
    numSliders = 4
    default_slider_names, default_slider_values = dh.get_default_slider_values(team, data)

    for i in range(numSliders):
        if default_slider_names[i] in int_stats:
            default_slider_values[i] = round(default_slider_values[i], 0)
        else:
            default_slider_values[i] = round(default_slider_values[i], 2)

    return default_slider_values[0], default_slider_values[1], default_slider_values[2], default_slider_values[3]

@app.callback(
    dash.dependencies.Output(component_id='slider-1', component_property= 'min'),
    dash.dependencies.Output(component_id='slider-2', component_property= 'min'),
    dash.dependencies.Output(component_id='slider-3', component_property= 'min'),
    dash.dependencies.Output(component_id='slider-4', component_property= 'min'),
    dash.dependencies.Output(component_id='slider-1', component_property= 'max'),
    dash.dependencies.Output(component_id='slider-2', component_property= 'max'),
    dash.dependencies.Output(component_id='slider-3', component_property= 'max'),
    dash.dependencies.Output(component_id='slider-4', component_property= 'max'),
    dash.dependencies.Output(component_id='slider-1', component_property= 'step'),
    dash.dependencies.Output(component_id='slider-2', component_property= 'step'),
    dash.dependencies.Output(component_id='slider-3', component_property= 'step'),
    dash.dependencies.Output(component_id='slider-4', component_property= 'step'),
    [dash.dependencies.Input(component_id='team_select', component_property='value')],
)

def update_slider_min_max_step(team):
    ''' Sets the min, max, and step of the stat sliders when a new team is chosen. References float_stats and int_stats '''  
    numSliders = 4

    slider_mins = []
    slider_maxes = []
    slider_steps = []

    default_slider_names, _ = dh.get_default_slider_values(team, data)

    for i in range(numSliders):
        if default_slider_names[i] in float_stats:
            slider_mins.append(0)
            slider_maxes.append(1)
            slider_steps.append(0.01)
        else:
            slider_mins.append(0)
            slider_maxes.append(100)
            slider_steps.append(1)

    return slider_mins[0], slider_mins[1], slider_mins[2], slider_mins[3], slider_maxes[0], slider_maxes[1], slider_maxes[2], slider_maxes[3], slider_steps[0], slider_steps[1], slider_steps[2], slider_steps[3]


if __name__ == '__main__':
    app.run_server(debug=True,  port= 8050)