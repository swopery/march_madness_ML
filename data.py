import numpy as np
import csv
from operator import add
from operator import mul

def load_data(year):
    """
    params: year

    Returns training_data, validation_data, cumalative_stats, t_data, t_data_formatted, tourney_data, teams
    """
    teams = load_teams()

    if year > 2016:
        season_data = load('RegularSeasonDetailedResults_Prelim2018.csv')
        tourney_data = load('NCAATourneyDetailedResults.csv')
    else:
        season_data = load('RegularSeasonDetailedResults.csv')
        tourney_data = load('TourneyDetailedResults.csv')

    t_data = load_year(year, tourney_data)
    t_data_formatted = format_game_data(t_data)

    # Need to reload due to deleted element, creates array dimension mismatch
    tourney_data = load('NCAATourneyDetailedResults.csv')

    train_data, cumalative_stats = format_game_data(season_data, year)
    split = int(len(train_data) * .2)

    # make sure even length
    if split%2 != 0:
        split -= 1

    validation_data = train_data[:split]
    training_data = train_data[split:]

    return training_data, validation_data, cumalative_stats, t_data, t_data_formatted, tourney_data, teams

def load_year(year, data):
    """
    params: year, data

    Returns subset of given array of tournament data for given year
    """
    tourney_games = []
    for game in data:
        if game[0] == str(year):
            tourney_games.append(game)
    return tourney_games

def load(csv_file):
    """
    params: csv_file

    Returns array from loading data row by row from input csv file into array
    """
    with open(csv_file, 'rt') as csvfile:
        data = csv.reader(csvfile)
        data_array = []
        next(data) #skip column titles
        for row in data:
            data_array.append(row)
        return data_array

def load_teams():
    """
    Returns dictionary of {ID : Team Name}
    """
    with open('Teams.csv', 'rt') as csvfile:
        data = csv.reader(csvfile)
        teams = {}
        next(data) #skip column titles 
        for row in data:
            teams[int(row[0])] = row[1]
        return teams

def format_game_data(game_data, year=None):
    """
    params: game_data, year(optional)

    Returns labeled training data (train_data).
    If year param is provided, also returns avg cumalative stats (team_dict)

    Train data is a tuple that contains a (list of stats, label 1 or 0 for win/loss)

    Dictionary team_dict stores team stats with key as team id and value as tuple.
    The tuple is as follows : (list of cumulative stats, number of games played)

    TRAIN ON ALL DATA, BUT ONLY GENERATE CUM STATS FROM WITHIN 4 YEAR BEFORE TOURNAMENT

    Sample cumalative stats array
    [ AVG_SCORE, 
    AVG_FGM, 
    AVG_FGA, 
    AVG_FGM3, 
    AVG_FGA3, 
    AVG_FTM, 
    AVG_FTA, 
    AVG_OR, 
    AVG_DR,
    AVG_AST, 
    AVG_TO, 
    AVG_STL,
    AVG_BLK, 
    AVG_PF ]
    """
    train_data = []
    team_dict = {}

    for game in game_data:
        # Remove unwanted data feature (number of overtime periods)
        del(game[7])

        winning_team_id = int(game[2])
        losing_team_id = int(game[4])
        game_year = int(game[0])
        
        winning_stats = game[7:20]
        losing_stats = game[20:33]
        
        # insert winning and losing score
        winning_stats.insert(0, game[3])
        losing_stats.insert(0, game[5])

        winning_stats = list(map(int, winning_stats))
        losing_stats = list(map(int, losing_stats))
        win_np = np.array(winning_stats)
        lose_np = np.array(losing_stats)
        # TODO: TRY VARIATIONS ON STATS (PERCENTAGES)
        # TODO: MATCHUPS VS ISOLATED STATS

        # Add win/lose label to NumPy arrays (1/0)
        train_data_win = (np.reshape(win_np, (len(win_np), 1)), 1)
        train_data_lose = (np.reshape(lose_np, (len(lose_np), 1)), 0)

        train_data.append(train_data_win)
        train_data.append(train_data_lose)

        # TODO: TRY VARIATIONS ON YEAR WINDOW FOR STATS
        if (year and 0 <= year - game_year <= 4):

            if winning_team_id not in team_dict:
                team_dict[winning_team_id] = (winning_stats, 1)

            if losing_team_id not in team_dict:
                team_dict[losing_team_id] = (losing_stats, 1)

            if winning_team_id in team_dict:
                winning_holder = list(team_dict[winning_team_id][0])
                winning_holder = list(map(add, winning_holder, winning_stats))
                games_played = team_dict[winning_team_id][1]+1
                team_dict[winning_team_id] = (winning_holder, games_played)

            if losing_team_id in team_dict:
                losing_holder = list(team_dict[losing_team_id][0])
                losing_holder = list(map(add, losing_holder, losing_stats))
                games_played = team_dict[losing_team_id][1]+1
                team_dict[losing_team_id] = (losing_holder, games_played)

    for team in team_dict:
        cum_stats = list(team_dict[team][0])
        games_played = team_dict[team][1]
        cum_stats = [stat/games_played for stat in cum_stats]
        team_dict[team] = (cum_stats, games_played)

    if (year):
        return train_data, team_dict
    else:
        return train_data
