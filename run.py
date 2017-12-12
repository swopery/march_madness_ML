# run NN code

from data import load_data
import nnet
import numpy as np
import math
from sklearn import metrics

# [input, hidden layer, output]
net = nnet.Network([14, 500, 2])

def main():
    # Main function prompts user to enter a year, trains and gathers cumulative
    # stats from the 4 years leading up to that year, and prints the following:

    # 1) Predicted outcomes in games of tournament
    # 2) Log loss associated with predictions
    # 3) Accuracy of predicted tournament outcomes
    # 4) Accuracy of predicted first round games
    # 5) Accuracy of model on all tournaments in dataset (2003-2016)

    input_year = int(input("Please enter year (2003-2016): ").strip())

    if not (2003 <= input_year <= 2016):
        print("Error, please enter year in correct range")
        print("Exiting...")
        return

    print("loading dataset")
    train_data, valid_data, cum_stats, t_data, t_data_formatted, tournament_data, teams = load_data(input_year)

    print("training")
    net.train(train_data, valid_data, epochs=20, mini_batch_size=10, alpha=0.5)

    # Full tournament predictions and log loss
    evaluate_tournament(t_data, cum_stats, teams, input_year)
    # Accuracy on tournament
    print(str(input_year) + " tournament ", end="")
    test_tournament(t_data, cum_stats)
    # Accuracy on first round
    first_round(t_data, cum_stats, teams)
    print("All tournaments (2003-2016) ", end="")
    # Accuracy on all tournaments (2003-2016)
    test_tournament(tournament_data, cum_stats)

def evaluate_tournament(tournament_data, cum_stats, teams, year):
    print('')
    print("******************************")
    print(str(year) + " TOURNAMENT PREDICTED RESULTS")
    print("******************************")
    log_loss = []
    actual_loss = []

    for game in tournament_data:

        team1, team2 = teams[int(game[2])], teams[int(game[4])]
        output1, output2, log_loss_input = predict_game(game, cum_stats)

        log_loss.append(log_loss_input)
        actual_loss.append(output1)

        output1 = str(output1).replace("1", "Win").replace("0", "Loss")
        output2 =  str(output2).replace("1", "Win").replace("0", "Loss")

        print(team1 + ": " + output1)
        print(team2 + ": " + output2)
        print("******************************")

    final = metrics.log_loss(actual_loss, log_loss)
    print("Log Loss: ", final)

def predict_game(game, cum_stats):
        team1_stats = cum_stats[int(game[2])][0]
        team2_stats = cum_stats[int(game[4])][0]

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

        return output1, output2, log_loss_input

def test_tournament(tournament_data, cum_stats):
    test_data = []
    for game in tournament_data:
        team1 = int(game[2])
        team2 = int(game[4])
        win_stats = cum_stats[team1][0]
        lose_stats = cum_stats[team2][0]

        win_np = np.array(win_stats)
        lose_np = np.array(lose_stats)

        test_data_win = (np.reshape(win_np, (len(win_np), 1)), 1)
        test_data_lose = (np.reshape(lose_np, (len(lose_np), 1)), 0)
        test_data.append(test_data_win)
        test_data.append(test_data_lose)

    tr, ntr = net.evaluate(test_data), len(test_data)
    print("accuracy %d/%d (%.2f%%) \n" % (tr, ntr, 100*tr/ntr), end='')

def first_round(tournament_data, cum_stats, teams):
    # Calculate the first round accuracy (x/32) of our model for input year

    # First 4 games will be play-ins, not truly "first round" like next 32
    first_round = tournament_data[4:36]
    second_round = []
    for game in first_round:
        team1 = teams[int(game[2])]
        team2 = teams[int(game[4])]
        output1, output2, temp = predict_game(game, cum_stats)

        if output1 == 1:
            second_round.append(team1)

    first, second = len(first_round), len(second_round)
    print("Round of 64 accuracy: %d/%d (%.2f%%) \n" % (second, first, 100*second/first), end='')

if __name__ == '__main__':
    main()

