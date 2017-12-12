import numpy as np
import csv
from operator import add
from operator import mul

def load_data(year):
    teams = load_teams()
    season_data = load('RegularSeasonDetailedResults.csv')
    tourney_data = load('TourneyDetailedResults.csv')

    t_data = load_year(year, tourney_data)
    t_data_formatted = format_game_data(load_year(year, tourney_data))

    # Need to reload due to deleted element, creates array dimension mismatch
    tourney_data = load('TourneyDetailedResults.csv')

    train_data, cum_stats = format_game_data(season_data, year)
    split = int(len(train_data) * .2)

    # make sure even length
    if split%2 != 0:
        split -= 1

    valid_data = train_data[:split]
    train_data = train_data[split:]

    return train_data, valid_data, cum_stats, t_data, t_data_formatted, tourney_data, teams

def load_year(year, data):
    tourney_games = []
    for game in data:
        if game[0] == str(year):
            tourney_games.append(game)
    return tourney_games

def load(csv_file):
    with open(csv_file, 'rt') as csvfile:
        data = csv.reader(csvfile)
        data_array = []
        next(data)
        for row in data:
            data_array.append(row)
        return data_array

def load_teams():
    with open('Teams.csv', 'rt') as csvfile:
        data = csv.reader(csvfile)
        teams = {}
        next(data)
        for row in data:
            teams[int(row[0])] = row[1]
        return teams

def format_game_data(game_data, year=None):
    train_data = []
    team_dict = {}

    # Train data is a tuple that contains a (list of stats, label 1 or 0 for win/loss)

    # Dictionary team_dict stores team stats with key as team id and value as tuple.
    # The tuple is as follows : (list of cumulative stats, number of games played)
    # Function returns team_dict stats as averages by dividing by number of games

    # TRAIN ON ALL DATA, BUT ONLY GENERATE CUM STATS FROM WITHIN 4 YEAR BEFORE TOURNAMENT
    for game in game_data:
        game_year = int(game[0])

        # Remove unwanted data feature
        del(game[7])

        winning_team = int(game[2])
        losing_team = int(game[4])
        game_year = int(game[0])
        winning_stats = game[8:20]
        # insert winning score
        winning_stats.insert(0, game[3])
        losing_stats = game[21:33]
        # insert losing score
        losing_stats.insert(0, game[5])

        winning_stats = list(map(int, winning_stats))
        losing_stats = list(map(int, losing_stats))

        winner_efficiency = (winning_stats[0]+ winning_stats[7]+ winning_stats[8]+ winning_stats[9]+ winning_stats[11]+ winning_stats[12]- (winning_stats[3]-winning_stats[2])-(winning_stats[6]-winning_stats[5])-winning_stats[10])/4
        winning_stats.append(int(winner_efficiency))

        losing_stats = game[21:33]
        # insert losing score
        losing_stats.insert(0, game[5])
        losing_stats = list(map(int, losing_stats))

        loser_efficiency = (losing_stats[0]+ losing_stats[7]+ losing_stats[8]+ losing_stats[9]+ losing_stats[11]+ losing_stats[12]- (losing_stats[3]-losing_stats[2])-(losing_stats[6]-losing_stats[5])-losing_stats[10])/4
        losing_stats.append(int(loser_efficiency))

        win_np = np.array(winning_stats)
        lose_np = np.array(losing_stats)

        train_data_win = (np.reshape(win_np, (len(win_np), 1)), 1)
        train_data_lose = (np.reshape(lose_np, (len(lose_np), 1)), 0)
        train_data.append(train_data_win)
        train_data.append(train_data_lose)

        if (year and 0 <= year - game_year <= 4):

            if winning_team not in team_dict:
                team_dict[winning_team] = (winning_stats, 1)

            if losing_team not in team_dict:
                team_dict[losing_team] = (losing_stats, 1)

            if winning_team in team_dict:
                winning_holder = list(team_dict[winning_team][0])
                winning_holder = list(map(add, winning_holder, winning_stats))
                games_played = team_dict[winning_team][1]+1
                team_dict[winning_team] = (winning_holder, games_played)

            if losing_team in team_dict:
                losing_holder = list(team_dict[losing_team][0])
                losing_holder = list(map(add, losing_holder, losing_stats))
                games_played = team_dict[losing_team][1]+1
                team_dict[losing_team] = (losing_holder, games_played)

    for team in team_dict:
        cum_stats = list(team_dict[team][0])
        games_played = team_dict[team][1]
        cum_stats = [stat/games_played for stat in cum_stats]
        team_dict[team] = (cum_stats, games_played)

    if (year):
        return train_data, team_dict
    else:
        return train_data

