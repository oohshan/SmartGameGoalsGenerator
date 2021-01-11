#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np


# In[89]:


import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)


# In[90]:


df = pd.read_csv("team_boxscores_v1.csv")


# In[91]:


# Kentucky
team1 = '2267a1f4-68f6-418b-aaf6-2aa0c4b291f1'


# In[92]:


# Florida
team2 = '912f8837-1d81-4ef9-a576-a21f271d4c64'


# In[93]:


# Returns all records for a given teamID
def getAllTeamMatchRecords(teamID, df):
    return df[df["team_id"] == teamID]


# In[94]:


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
print(statWinLoss(team2, df))


# In[95]:


def getGameIDs(teamID, df):
    return df[df["team_id"] == teamID]["game_id"]


# In[96]:


# Returns common game IDs between two teams
def getMatchupGameIDs(team1, team2, df):
    return pd.merge(getGameIDs(team1, df), getGameIDs(team2, df))


# In[98]:


# Returns average of a given statistic for a given team
def getAvgStatForTeam(teamID, statistic, df):
    runningSum = 0
    runningCount = 0
    
    team_stats = df[df["team_id"] == teamID]
    for index, row in team_stats.iterrows():
        runningSum += row[statistic]
        runningCount += 1
    
    return runningSum / runningCount


# In[111]:


# Gets stats between two teams (UK vs LSU w/ Respect to UK)
matchupStats = pd.merge(getMatchupGameIDs(team1, team2, df), getAllTeamMatchRecords(team1, df))

# Gets wins and losses between team1 and team2 and returns seperate dataframes
def testey(team1, team2, matchupStats):
    wins = pd.DataFrame()
    losses = pd.DataFrame()
    
    for index, row in matchupStats.iterrows():
        if row["points"] > row["points_against"]:
            wins = wins.append(matchupStats.iloc[index])
        else:
            losses = losses.append(matchupStats.iloc[index])
    return wins, losses

wins, losses = testey(team1, team2, matchupStats)

avgWinAssists = wins["assists"].mean()
avgLossAssists = losses["assists"].mean()
print(avgWinAssists, avgLossAssists)


# In[ ]:




