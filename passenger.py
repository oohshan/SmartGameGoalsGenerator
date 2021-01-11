#!/usr/bin/env python
# coding: utf-8

# In[37]:


# Questions for 10/28 meeting:
# Test set  -> Should the test be just one game? Answer: Leave it the way it is for now.
# Train set -> Should we duplicate previous games to add weighting? Answer: Yes.

## November 6th, 2020 Backend Meeting ##
# 4 Factors to include for opponent: efg, tov_pct, orb_pct, ftr ... - Done
# Add win (boolean) column for each game -> predict on that instead of points - Done
# Later on: Using most recent games???

## November 10th, 2020 Backend Meeting ##
# Next Steps:
# Get it on the dashboard
# Other functionality?

# Imports
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from matplotlib import pyplot
pd.set_option("display.max_rows", None, "display.max_columns", None)


# In[38]:


# Read in box score data provided by Ludis
df = pd.read_csv("team_boxscores_v3.csv")
df = df.fillna(0)

# pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', 59)


# In[39]:


### Hard-coded teamIDs from dataset for testing purposes ###

# Kentucky
team1 = '2267a1f4-68f6-418b-aaf6-2aa0c4b291f1'

# LSU
team2 = '70e2bedd-3a0a-479c-ac99-e3f58aa6824b'

# Ohio State
team3 = '857462b3-0ab6-4d26-9669-10ca354e382b'

# Florida
team4 = '912f8837-1d81-4ef9-a576-a21f271d4c64'

# Michigan State
team5 = 'a41d5a05-4c11-4171-a57e-e7a1ea325a6d'

floatArr = ["efg","orb_pct","ftr"]
negFloatArr = ["tov_pct"]
intArr = ["assists", "blocks","defensive_rebounds", "fast_break_pts", "points_in_paint","points_off_turnovers","rebounds","steals"]
negIntArr = ["turnovers","opponent_drb"]


# In[40]:


# Returns all game records for a given teamID
def getAllTeamMatchRecords(teamID, df):
    return df[df["team_id"] == teamID]


# In[41]:


# Returns win/loss ratio for a given team across entire dataset
# Add functionality for filtering by season?
def statWinLoss(teamID, df):
    wins = 0
    losses = 0
    team_stats = df[df["team_id"] == teamID]
    for index, row in team_stats.iterrows():
        if row["points"] > row["points_against"]:
            wins = wins + 1
        else:
            losses = losses + 1
    if losses == 0:
        return 1
    else:
        return wins/losses


# In[42]:


# Return all gameIDs for a given team
def getGameIDs(teamID, df):
    return df[df["team_id"] == teamID]["game_id"]


# In[43]:


# Returns common game IDs between two teams
def getMatchupGameIDs(team1, team2, df):
    return pd.merge(getGameIDs(team1, df), getGameIDs(team2, df))


# In[44]:


# Returns average of a given statistic for a given teamID
def getAvgStatForTeam(teamID, statistic, df):
        runningSum = 0
        #runningSum = float(0)
        runningCount = 0
        team_stats = df[df["team_id"] == teamID]
        for index, row in team_stats.iterrows():
            runningSum += row[statistic]
            runningCount += 1
         
            return runningSum / runningCount
            return runningSum / runningCount
     
        print(getAvgStatForTeam(team1, "rebounds", df))


# In[45]:


# This function will get the record of a team by a specific year and can also calculate some avg
def getTeamRecordByYear(teamID, year, df):
    team_record = df[df["team_id"] == teamID]
    sum_two_pts_made = 0
    count = 0
    avg_two_pts_made = 0
    sum_field_goals_made =0
    count2 = 0
    avg_field_goals_made = 0
    for index, row in team_record.iterrows():
        if (row["season"] == year): 
           team_record1 = team_record[df["season"] == row["season"]]
           for index, row in team_record1.iterrows():
                sum_two_pts_made += row["two_points_made"]
                sum_field_goals_made += row["field_goals_made"]
                count +=1
                count2 +=1
           avg_two_pts_made = sum_two_pts_made / count
           avg_field_goals_made = sum_field_goals_made / count2
           return_value = "%f %f" %(avg_two_pts_made,avg_field_goals_made)
           return team_record1


# In[46]:


# Return dataframe with selected features
def filterRowsFS(df):
    return df[["assists","blocks","defensive_rebounds","opponent_drb","fast_break_pts","points_in_paint","points_off_turnovers","rebounds","steals","turnovers","efg","tov_pct","orb_pct","ftr"]]


# In[105]:


# Calculate correct predictions -> wins/losses
def calcPredError(df):
    error = 0
    correct = 0
    i = 0
    for index, row in df.iterrows():
        i = i + 1
        if df.loc[index, 'Actual'] != df.loc[index, 'Predicted (int)']:
            error = error + 1
        else:
            correct = correct + 1
    return ((correct / i) * 100)


# In[48]:


# Calculate win percentage
def winPct(teamPred):
    # return round((teamPred['Predicted (float)'].sum() / len(teamPred['Predicted (float)']) * 100))
    return float(teamPred['Predicted (float)'].sum() / len(teamPred['Predicted (float)']) * 100)


# In[49]:


# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# In[50]:


def overallFeatures(df):
    datasetForFS = df
    datasetForFS.fillna(0)

    # X1 = datasetForFS[["assists","personal_fouls","ftr","orb_pct", "tov_pct", "points_in_paint", "blocks"]]
    # X1 = datasetForFS[["assists","blocks","personal_fouls"]]
    X1 = datasetForFS[["assists","blocks","defensive_rebounds","opponent_drb","fast_break_pts","points_in_paint","points_off_turnovers","rebounds","steals","turnovers","efg","tov_pct","orb_pct","ftr"]]
    y1 = datasetForFS['win']

    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=0)
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

    colList = X1.columns.values.tolist()
    statScoreDF = pd.DataFrame(data={'Stat': pd.Series(colList), 'Score': pd.Series(fs.scores_.tolist())})
    statScoreDF = statScoreDF.sort_values(by=['Score'], ascending=False)

    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
    
    return statScoreDF

# print(overallFeatures(df))


# In[122]:


def teamFeatures(team1, team2, df):
    datasetForFS = getAllTeamMatchRecords(team1, df).merge(getMatchupGameIDs(team1, team2, df))
    datasetForFS.fillna(0)

    # X1 = datasetForFS[["assists","personal_fouls","ftr","orb_pct", "tov_pct", "points_in_paint", "blocks"]]
    # X1 = datasetForFS[["assists","blocks","personal_fouls"]]
    X1 = datasetForFS[["assists","blocks","defensive_rebounds","opponent_drb","fast_break_pts","points_in_paint","points_off_turnovers","rebounds","steals","turnovers","efg","tov_pct","orb_pct","ftr"]]
    y1 = datasetForFS['win']

    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=0)
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

    colList = X1.columns.values.tolist()
    statScoreDF = pd.DataFrame(data={'Stat': pd.Series(colList), 'Score': pd.Series(fs.scores_.tolist())})
    statScoreDF = statScoreDF.sort_values(by=['Score'], ascending=False)

    # Plot the scores - PyPlot
    # pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    # pyplot.show()
    
    return statScoreDF

# teamFeatures(team1, team2, df)


# In[123]:


def learn(dataset):
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
    
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    return regressor, pd.DataFrame({'Actual': y_test, 'Predicted (int)': y_pred_round, 'Predicted (float)': y_pred})

# reg, pred = learn(pd.read_csv("team_boxscores_v3.csv"))
# print(calcPredError(pred), winPct(pred))

# df1 = filterRowsFS(getAllTeamMatchRecords(team1, df))
# df2 = getAllTeamMatchRecords(team1, df)["win"]
# dfPred = reg.predict(df1)
# dfPredRound = np.around(dfPred)

# temp = pd.DataFrame({'Actual': df2, 'Predicted (int)': dfPredRound, 'Predicted (float)': dfPred})

# print(calcPredError(temp), winPct(temp))


# In[124]:


def learnMatchup(team1, team2):
    dataset = pd.read_csv("team_boxscores_v3.csv")
    dataset = dataset.fillna(0)
    dfTeam1 = getAllTeamMatchRecords(team1, dataset)
    matchups = getMatchupGameIDs(team1, team2, df)["game_id"].tolist()
    dfTeam1 = dfTeam1.reset_index()
    
    # Elijah - Save rows for later and append to train set
    for index, row in dfTeam1.iterrows():
        for i in range(0, len(matchups)):
            if str(dfTeam1.loc[index, "game_id"]) == matchups[i]:
                dfTeam1 = dfTeam1.append(dfTeam1.loc[index], ignore_index=True)
    
    dfTeam1 = dfTeam1.sample(frac = 1) 
                
    X1 = dfTeam1[["assists","blocks","defensive_rebounds","opponent_drb","fast_break_pts","points_in_paint","points_off_turnovers","rebounds","steals","turnovers","efg","tov_pct","orb_pct","ftr"]]
    y1 = dfTeam1['win']
    
    # rng = np.random.randint(0, 42)
    rng = 0
    # X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=rng)
    
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
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    return regressor, pd.DataFrame({'Actual': y_test, 'Predicted (int)': y_pred_round, 'Predicted (float)': y_pred})

reg, pred = learnMatchup(team1, team2)


# In[125]:


def avgDataRow(df):
    df1 = dict()
    for (columnName, columnData) in df.iteritems():
        df1[columnName] = [df[columnName].mean()]
    
    return pd.DataFrame(df1)


# In[128]:


stats = teamFeatures(team1, team2, df).head()['Stat'].tolist()

df1 = getAllTeamMatchRecords(team1, df)
df2 = avgDataRow(filterRowsFS(getAllTeamMatchRecords(team1, df)))
df3 = df1["win"]


dfPred = reg.predict(df2)
dfPredRound = np.around(dfPred)
dfFinal = pd.DataFrame({'Actual': df3.mean(), 'Predicted (int)': dfPredRound, 'Predicted (float)': dfPred})
print(dfFinal)

# print(df2)
df2.at[0,"assists"] = df2.at[0,"assists"] + 10
dfPred = reg.predict(df2)
dfPredRound = np.around(dfPred)
dfFinal = pd.DataFrame({'Actual': df3.mean(), 'Predicted (int)': dfPredRound, 'Predicted (float)': dfPred})
print(dfFinal)
# print(df2)


# In[54]:


# Return win percentage as stat changes
# df - dataframe, e.g. getAllTeamMatchRecords(team1, df)
# reg - regressor from above
# var - the feature to change
# val - the value to add to the feature
def predOnStat(df, reg, var, val):
    df1 = df[["assists","blocks","defensive_rebounds","opponent_drb","fast_break_pts","points_in_paint","points_off_turnovers","rebounds","steals","turnovers","efg","tov_pct","orb_pct","ftr"]]
    for index, row in df1.iterrows():
        df1.at[index, var] = df1.at[index, var] + val
    
    temp_pred = reg.predict(df1)
    temp_pred_round = np.around(reg.predict(df1))
    
    test = pd.DataFrame({'Actual': df['win'], 'Predicted (int)': temp_pred_round, 'Predicted (float)': temp_pred})
    return float(winPct(test))


# In[ ]:


# df  -> dataframe
# reg -> regressor
# Return new win pct
def updateWinPct(df, reg):
    reg.predict()
    


# In[28]:


# statList = ["assists", "blocks", "orb_pct"]
def compTeams(df, teamID, opponentID, win_percent):
    topFive = teamFeatures(teamID, opponentID, df)["Stat"].head().tolist()
    print(topFive)
    reg, pred = learnMatchup(teamID, opponentID)
    
    intVal = 0
    floatVal = 0
    originalPct = predOnStat(getAllTeamMatchRecords(teamID, df), reg, 'assists', 0)
    
    for stat in topFive:
        currentPct = originalPct
        print(stat)
        floatVal = 0
        intVal = 0
        if stat in intArr:
            while (currentPct <= win_percent):
                print("intyyy")
                intVal = intVal + 1
                currentPct = predOnStat(getAllTeamMatchRecords(teamID, df), reg, stat, intVal)
            print(stat, intVal)
        
        if stat in negIntArr:
            while (currentPct <= win_percent):
                print("neggggintyyy")
                intVal = intVal - 1
                currentPct = predOnStat(getAllTeamMatchRecords(teamID, df), reg, stat, intVal)
            print(stat, intVal)
        
        elif stat in floatArr:
            while (currentPct <= win_percent):
                print("floattty")
                floatVal = floatVal + 0.1
                currentPct = predOnStat(getAllTeamMatchRecords(teamID, df), reg, stat, floatVal)
            print(stat, floatVal)
        
        elif stat in negFloatArr:
            while (currentPct <= win_percent):
                print("neggggfloattty")
                floatVal = floatVal - 0.1
                currentPct = predOnStat(getAllTeamMatchRecords(teamID, df), reg, stat, floatVal)
            print(stat, floatVal)

    print(val)
    return temp

win_percent = 80.5
compTeams(df, team1, team2, win_percent)


# In[19]:


# testey = getAllTeamMatchRecords(team1, df)
# prediction_acc, win_percent = predOnStat(testey, reg, "assists", 0)
# print("Prediction accuracy:", prediction_acc, "\nWin Percent:", win_percent)
# prediction_acc, win_percent = predOnStat(testey, reg, "assists", 5)
# print("Prediction accuracy:", prediction_acc, "\nWin Percent:", win_percent)
# prediction_acc, win_percent = predOnStat(testey, reg, "assists", 10)
# print("Prediction accuracy:", prediction_acc, "\nWin Percent:", win_percent)


# In[ ]:




