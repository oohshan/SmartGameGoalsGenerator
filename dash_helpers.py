import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime as dat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest

# TODO: 
#       The model doesn't really take the opponent into consideration when calculating the win percentage. As you would expect, this is not ideal and is something that needs to be fixed
# 
#       The bar charts only graph 2019 data. Allowing the user to choose the year would be an easy addition. Future implemenatations could also include a previous X games option instead
# 
#       The bar chart only graphs stats correctly if they are selected in order. For example, if the set of possible stats are ['assists', 'rebounds', 'blocks'], they must all be selected
#       in order to show all the values correctly. If the user selects only 'assists' and 'blocks' then 'assists' graphs correctly. 'blocks' is graphed but is given the value assigned
#       to 'rebounds' because it assumes the second position in the array of stats to be graphed. 
#
#       The model doesn't run well (and generally fails) for small schools due to a lack of data for those teams. Error checking needs to be implemented to eliminate this problem.


def getStatsByYear(teamID, year, data):
    ''' Returns the stats for a chosen team for a specific year. Choices are 2016 - 2019 '''
    teamStats = data[data["team_id"] == teamID]
    
    for index, row in teamStats.iterrows():
        if (row["season"] == year): 
           teamStatsForGivenYear = teamStats[data["season"] == row["season"]]
           return teamStatsForGivenYear

def generate_bar_chart(team, opponent, stats, stat_names, data):
    ''' Generates a bar chart for a the user selected team, opponent and stats ''' 

    teamStats = getStatsByYear(team, 2019, data)
    opponentStats = getStatsByYear(opponent, 2019, data)

    teamStatValues = teamStats[["assists", "assists_turnover_ratio", "blocked_att", "blocks", "defensive_rebounds", "fast_break_pts",
                                "field_goals_att", "field_goals_pct", "field_goals_made", "free_throws_att",
                                "free_throws_pct", "free_throws_made", "offensive_rebounds", "personal_fouls", 
                                "points", "points_against", "points_in_paint", "points_off_turnovers",
                                "rebounds", "second_chance_pts", "steals", "team_rebounds", "three_points_att",
                                "three_points_pct", "three_points_made", "turnovers", "two_points_att",
                                "two_points_pct", "two_points_made"
                                ]]

    opponentStatValues = opponentStats[["assists", "assists_turnover_ratio", "blocked_att", "blocks", "defensive_rebounds", "fast_break_pts",
                                "field_goals_att", "field_goals_pct", "field_goals_made", "free_throws_att",
                                "free_throws_pct", "free_throws_made", "offensive_rebounds", "personal_fouls", 
                                "points", "points_against", "points_in_paint", "points_off_turnovers",
                                "rebounds", "second_chance_pts", "steals", "team_rebounds", "three_points_att",
                                "three_points_pct", "three_points_made", "turnovers", "two_points_att",
                                "two_points_pct", "two_points_made"
                                ]]

    stats_to_be_graphed = []

    for i in range(len(stat_names)):
        if i in stats:
            stats_to_be_graphed.append(stat_names[i])

    # Graphs average stat values for the user's chosen team
    teamVals = go.Bar(
        x = stats_to_be_graphed,
        y = teamStatValues.mean(),
        name = data[(data.team_id == team)]['market'].iloc[0]
    )

    # Graphs average stat values for the opponent's team
    opponentVals = go.Bar(
        x = stats_to_be_graphed,
        y = opponentStatValues.mean(),
        name = data[(data.team_id == opponent)]['market'].iloc[0]
    )
    
    data = [teamVals, opponentVals]
    layout = go.Layout(barmode = 'group')
    fig = go.Figure(data = data, layout = layout)
    
    return fig

def getAllTeamMatchRecords(teamID, df):
    ''' Returns all game records for a given teamID '''
    return df[df["team_id"] == teamID]

def select_features(X_train, y_train, X_test):
    ''' Selects features '''
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def overallFeatures(df):
    ''' Return list of top four features '''
    datasetForFS = df
    datasetForFS.fillna(0)

    X1 = datasetForFS[["assists","blocks","defensive_rebounds","opponent_drb","fast_break_pts","points_in_paint","points_off_turnovers","rebounds","steals","turnovers","efg","tov_pct","orb_pct","ftr"]]
    y1 = datasetForFS['win']

    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=0)
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

    colList = X1.columns.values.tolist()
    statScoreDF = pd.DataFrame(data={'Stat': pd.Series(colList), 'Score': pd.Series(fs.scores_.tolist())})
    statScoreDF = statScoreDF.sort_values(by=['Score'], ascending=False)
    
    return statScoreDF.head(n=4)['Stat'].tolist()

def avgDataRow(df):
    ''' Returns the average values of a dataFrame '''
    df1 = dict()
    for (columnName, columnData) in df.iteritems():
        df1[columnName] = [df[columnName].mean()]
    
    return pd.DataFrame(df1)

def updateWinPct(dfMain, dfWin, reg):
    ''' Return new win percentage '''
    dfPred = reg.predict(dfMain)
    return pd.DataFrame({'Actual': dfWin.mean(), 'Predicted (int)': np.around(dfPred), 'Predicted (float)': dfPred})

def filterRowsFS(df):
    ''' Return dataframe with selected features '''
    return df[["assists","blocks","defensive_rebounds","opponent_drb","fast_break_pts","points_in_paint","points_off_turnovers","rebounds","steals","turnovers","efg","tov_pct","orb_pct","ftr"]]

def learn(dataset):
    ''' Trains the model '''
    dataset = pd.read_csv("team_boxscores_v3.csv")
    dataset = dataset.fillna(0)
    
    # Shuffle
    dataset = dataset.sample(frac = 1) 
    
    X1 = dataset[["assists","blocks","defensive_rebounds","opponent_drb","fast_break_pts","points_in_paint","points_off_turnovers","rebounds","steals","turnovers","efg","tov_pct","orb_pct","ftr"]]
    y1 = dataset['win']
    
    # No shuffle
    # X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=0)
    
    # W/ shuffle
    X_train = X1[int(len(X1)/5):]
    X_test = X1[:int(len(X1)/5)]
    
    y_train = y1[int(len(y1)/5):]
    y_test = y1[:int(len(y1)/5)]
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    coeff_df = pd.DataFrame(regressor.coef_, X1.columns, columns=['Coefficient'])
    
    y_pred = regressor.predict(X_test)
    y_pred_round = np.around(regressor.predict(X_test))
    
    return regressor, pd.DataFrame({'Actual': y_test, 'Predicted (int)': y_pred_round, 'Predicted (float)': y_pred})

def calculate_win_percentage(team, stat1, stat2, stat3, stat4, regressor, data):
    ''' Calculates the win percentage for a team and the 4 selected stat values '''
    temp = getAllTeamMatchRecords(team, data)
    changed_stat1 = overallFeatures(temp)[0]
    changed_stat2 = overallFeatures(temp)[1]
    changed_stat3 = overallFeatures(temp)[2]
    changed_stat4 = overallFeatures(temp)[3]
    average_team_stats = avgDataRow(filterRowsFS(temp))
    dfWin = temp["win"]

    dfFinal = pd.DataFrame({'Actual': dfWin.mean(), 'Predicted (int)': np.around(regressor.predict(average_team_stats)), 'Predicted (float)': regressor.predict(average_team_stats)})
    origWinPct = dfFinal.at[0, 'Predicted (float)']

    average_team_stats.at[0, changed_stat1] = stat1
    average_team_stats.at[0, changed_stat2] = stat2
    average_team_stats.at[0, changed_stat3] = stat3
    average_team_stats.at[0, changed_stat4] = stat4

    win_percentage = updateWinPct(average_team_stats, dfWin, regressor).at[0,'Predicted (float)']

    # Makes sure you can't have a win percentage of > 100% or < 0%
    if win_percentage > 1:
        win_percentage = 1
    elif win_percentage < 0:
        win_percentage = 0

    win_percentage = win_percentage * 100
    win_percentage = round(win_percentage, 2)

    win_percentage_text = "Projected Win Percentage: " + str(win_percentage) + "%"

    return win_percentage_text

def get_default_slider_values(team, data):
    ''' Gets the values the each of the 4 sliders should display. These values are what the model estimates the team will get in the matchup '''
    numSliders = 4
    stat_column_names = []
    stat_column_values = []

    estimated_stat_values = avgDataRow(filterRowsFS(getAllTeamMatchRecords(team, data)))

    for i in range(numSliders):
        stat_column_names.append(overallFeatures(getAllTeamMatchRecords(team, data))[i])
        stat_column_values.append(estimated_stat_values.at[0, stat_column_names[i]])

    return stat_column_names, stat_column_values