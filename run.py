# run NN code

from data import load_data
import nnet
import numpy as np
import math
from sklearn import metrics

# [input, hidden layer, output]
net = nnet.Network([14, 500, 2])

def main():
    # Prompts user to enter a year, trains and gathers cumulative stats from the 
    # 4 years leading up to that year, and prints the following:

    # 1) Predicted outcomes in games of tournament
    # 2) Log loss associated with predictions
    # 3) Accuracy of predicted tournament outcomes
    # 4) Accuracy of predicted first round games
    # 5) Accuracy of model on all tournaments in dataset (2003-2017)

    input_year = int(input("Please enter year (2003-2018): ").strip())

    if not (2003 <= input_year <= 2018):
        print("Error, please enter year in correct range")
        print("Exiting...")
        return

    print("loading dataset")
    training_data, validation_data, cumulative_stats, t_data, t_data_formatted, tournament_data, teams = load_data(input_year)
    print("training")

    # Uncomment to increase training speed for debugging
    # training_data = training_data[:1000]
    # validation_data = validation_data[:1000]

    net.train(training_data, validation_data, epochs=20, mini_batch_size=10, alpha=0.5)

    if input_year == 2018:
        evaluate_current(cumulative_stats, teams, input_year)
    else:
        # Full tournament predictions and log loss
        evaluate_tournament(t_data, cumulative_stats, teams, input_year)

        # Accuracy on tournament
        print(str(input_year) + " tournament ", end="")
        test_tournament(t_data, cumulative_stats)
        
        # Accuracy on first round
        first_round(t_data, cumulative_stats, teams, input_year)

def evaluate_tournament(tournament_data, cumulative_stats, teams, year):
    """
    params: tournament_data, cumalative_stats, teams, year

    Returns nothing. Primarily for displaying predicted tournament results for a given year,
    as well as the log loss for predictions.
    """
    print('')
    print("******************************")
    print(str(year) + " TOURNAMENT PREDICTED RESULTS")
    print("******************************")
    log_loss = []
    actual_loss = []

    for game in tournament_data:

        team1, team2 = teams[int(game[2])], teams[int(game[4])]
        output1, output2, log_loss_input = predict_game(int(game[2]), int(game[4]), cumulative_stats, year)

        log_loss.append(log_loss_input)
        actual_loss.append(output1)

        output1 = str(output1).replace("1", "Win").replace("0", "Loss")
        output2 =  str(output2).replace("1", "Win").replace("0", "Loss")

        print(team1 + ": " + output1)
        print(team2 + ": " + output2)
        print("******************************")

    final = metrics.log_loss(actual_loss, log_loss)
    print("Log Loss: ", final)

def predict_game(team1, team2, cumulative_stats, year):
        team1_stats = cumulative_stats[team1][0]
        team2_stats = cumulative_stats[team2][0]

        win_np = np.array(team1_stats)
        lose_np = np.array(team2_stats)

        test_data_win = (np.reshape(win_np, (len(win_np), 1)), 1)
        test_data_lose = (np.reshape(lose_np, (len(lose_np), 1)), 0)

        output1_prob = net.feedforward(test_data_win[0])
        output2_prob = net.feedforward(test_data_lose[0])

        output1 = np.argmax(output1_prob)
        output2 = np.argmax(output2_prob)
        log_loss_input = ((output1_prob[output1] - output2_prob[output2])/2) + .5

        # Fix "double win"/"double loss"
        if output1 == output2:
            prob_list = [output1_prob[output1], output2_prob[output1]]
            max_prob = np.argmax(prob_list)
            if max_prob == 0:
                if output1 == 1:
                    output2 = 0
                else:
                    output1 = 1
            else:
                if output2 == 1:
                    output1 = 0
                else:
                    output2 = 1
        print(output1, output2)
        return output1, output2, log_loss_input

def test_tournament(tournament_data, cumulative_stats):
    test_data = []
    for game in tournament_data:
        team1 = int(game[2])
        team2 = int(game[4])
        win_stats = cumulative_stats[team1][0]
        lose_stats = cumulative_stats[team2][0]

        win_np = np.array(win_stats)
        lose_np = np.array(lose_stats)

        test_data_win = (np.reshape(win_np, (len(win_np), 1)), 1)
        test_data_lose = (np.reshape(lose_np, (len(lose_np), 1)), 0)
        test_data.append(test_data_win)
        test_data.append(test_data_lose)

    tr, ntr = net.evaluate(test_data), len(test_data)
    print("accuracy %d/%d (%.2f%%) \n" % (tr, ntr, 100*tr/ntr), end='')

def first_round(tournament_data, cumulative_stats, teams, year):
    # Calculate the first round accuracy (x/32) of our model for input year

    # First 4 games will be play-ins, not truly "first round" like next 32
    first_round = tournament_data[4:36]
    second_round = []
    for game in first_round:
        team1 = teams[int(game[2])]
        team2 = teams[int(game[4])]
        output1, output2, temp = predict_game(int(game[2]), int(game[4]), cumulative_stats, year)

        if output1 == 1:
            second_round.append(team1)

    first, second = len(first_round), len(second_round)
    print("Round of 64 accuracy: %d/%d (%.2f%%) \n" % (second, first, 100*second/first), end='')

def evaluate_current(cumulative_stats, teams, year):
    # Ordered bracket for 2018 was not available, had to hardcode
    print('')
    print("**********************************")
    print(str(year) + " TOURNAMENT PREDICTED RESULTS")
    print("**********************************")
    log_loss = []
    actual_loss = []

    tournament = ['1438', '1420',
                    '1166', '1243',
                    '1246', '1172',
                    '1112', '1138',
                    '1274', '1260',
                    '1397', '1460',
                    '1305', '1400',
                    '1153', '1209',
                    '1462', '1411',
                    '1281', '1199',
                    '1326', '1377',
                    '1211', '1422',
                    '1222', '1361',
                    '1276', '1285',
                    '1401', '1344',
                    '1314', '1252',
                    '1437', '1347',
                    '1439', '1104',
                    '1452', '1293',
                    '1455', '1267',
                    '1196', '1382',
                    '1403', '1372',
                    '1116', '1139',
                    '1345', '1168',
                    '1242', '1335',
                    '1371', '1301',
                    '1155', '1308',
                    '1120', '1158',
                    '1395', '1393',
                    '1277', '1137',
                    '1348', '1328',
                    '1181', '1233']
    temp = []

    while (len(tournament) >1):
        #copy elements of tournament into temp list
        temp = list(tournament)

        for i in range(0, len(tournament), 2):
            # Grab two teams that are playing, predict result
            team1, team2 = teams[int(tournament[i])], teams[int(tournament[i+1])]
            output1, output2, log_loss_input = predict_game(int(tournament[i]), int(tournament[i+1]), cumulative_stats, year)

            # Remove losing team
            if output1 == 1:
                temp.remove(tournament[i+1])
            else:
                temp.remove(tournament[i])

            log_loss.append(log_loss_input)
            actual_loss.append(output1)

            output1 = str(output1).replace("1", "Win").replace("0", "Loss")
            output2 =  str(output2).replace("1", "Win").replace("0", "Loss")

            print(team1 + ": " + output1)
            print(team2 + ": " + output2)
            print("******************************")

        # copy updated elements of temp list into tournament list
        tournament = list(temp)

    final = metrics.log_loss(actual_loss, log_loss)
    print("Log Loss: ", final)

if __name__ == '__main__':
    main()

