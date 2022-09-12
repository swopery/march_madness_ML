import random, math
import numpy as np

class Network(object):

    def __init__(self, sizes, debug=False):
        """
        Construct a new neural net with layer sizes given.  For
        example, if sizes = [2, 3, 1] then it would be a three-layer
        network, with the first layer containing 2 neurons, the
        second layer 3 neurons, and the third layer 1 neuron.
        The biases and weights for the network are initialized randomly.
        If debug=True then repeatable "random" values are used.
        """
        self.sizes = sizes
        self.debug = debug

        # Use rand_mat(rows, columns, debug) to initialize weights and biases
        self.biases =  []
        self.weights = []
        for i in range(len(sizes)-1):
            self.weights.append(rand_mat(sizes[i+1], sizes[i], debug))
            self.biases.append(rand_mat(sizes[i+1], 1, debug))

    def feedforward(self, a):
        """Return the output of the network if a is input"""
        for i in range(1,len(self.sizes)):
            bias = self.biases[i-1]
            weight = self.weights[i-1]
            z = np.dot(weight,a) + bias
            a = sigmoid(z)
        return a

    def train(self, train_data, valid_data, epochs, mini_batch_size, alpha):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.
        """

        self.report_accuracy(0, train_data, valid_data)

        # vectorize outputs y in training data ("one hot" encoding)
        ny = self.sizes[-1]
        train_data_vec = [(x, unit(y, ny)) for x, y in train_data]

        m = len(train_data)
        for j in range(epochs):
            alpha = alpha/2
            if not self.debug:
                random.shuffle(train_data_vec)

        # divide train_data_vec into batches (lists of size
        # mini_batch_size), and call self.update_mini_batch on each
            split_array = []
            for i in range(0, len(train_data_vec), mini_batch_size):
                mini_batch = train_data_vec[i:i + mini_batch_size]
                split_array.append(mini_batch)

            for mini_batch in split_array:
                self.update_mini_batch(mini_batch, alpha)

            self.report_accuracy(j+1, train_data, valid_data)


    def update_mini_batch(self, mini_batch, alpha):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        mini_batch is a list of tuples (x, y), and alpha
        is the learning rate.
        """
        gradient_biases = [np.zeros(b.shape) for bias in self.biases]
        gradient_weights = [np.zeros(w.shape) for weight in self.weights]
        nL = len(self.sizes)
        m = len(mini_batch)

        for x, y in mini_batch:
            delta_biases, delta_weights = self.backprop(x, y)
            gradient_biases = [sum(x) for x in zip(delta_biases, gradient_biases)]
            gradient_weights = [sum(x) for x in zip(delta_weights, gradient_weights)]

        self.biases = [(bias - ((alpha/m) * gradient)) for bias, gradient in zip(self.biases, gradient_biases)]
        self.weights = [(weight - ((alpha/m) * gradient)) for weight, gradient in zip(self.weights, gradient_weights)]

    def backprop(self, x, y):
        """
        Return (gradient_biases, gradient_weights) representing the gradient of the cost
        function for a single training example (x, y)
        """
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        n = len(self.sizes) # number of layers

        # forward pass
        # same as feedforward, but store all a's and z's in lists
        a = [0] * n
        z = [0] * n
        a[0] = x # initial activation (z[0] is not used)
        for i in range(1, n): # 1 .. n-1
            bias = self.biases[i-1]
            weight = self.weights[i-1]
            z[i] = np.dot(w, a[i-1]) + b
            a[i] = sigmoid(z[i])

        # backward pass
        delta = [0] * n
        i = n-1 # index of last layer
        delta[i] = (a[i]-y) * sigmoid_grad(z[i])
        for i in range(n-2, 0, -1): # n-2 .. 1
            delta[i] = np.dot(np.transpose(self.weights[i]), delta[i+1])*sigmoid_grad(z[i])

        # compute gradients
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        for i in range(0, n-1):
           gradient_biases[i] = delta[i+1]
           gradient_weights[i] = np.dot(delta[i+1],np.transpose(a[i]))
        return (gradient_biases, gradient_weights)

    def evaluate(self, data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result.
        """
        correct = 0
        # go through data two at a time to get both teams from game
        for i in range(0, len(data), 2):
            x0 = data[i][0]
            y0 = data[i][1]
            prob0 = self.feedforward(x0)

            output0 = np.argmax(prob0)

            x1 = data[i+1][0]
            y1 = data[i+1][1]
            prob1 = self.feedforward(x1)

            output1 = np.argmax(prob1)

            # Logic for correcting "double win" and "double loss"
            if output0 == output1:
                prob_list = [prob0[output0], prob1[output0]]
                max_prob = np.argmax(prob_list)
                if max_prob == 0:
                    if output0 == 1:
                        output1 = 0
                    else:
                        output0 = 1
                else:
                    if output1 == 1:
                        output0 = 0
                    else:
                        output1 = 1

            # now that outputs are not equal, if one is right, the other is rights
            if output0 == y0:
                    correct += 2
        return correct

    def report_accuracy(self, epoch, train_data, valid_data=None):
        """report current accuracy on training and validation data"""
        tr, ntr = self.evaluate(train_data), len(train_data)
        te, nte = self.evaluate(valid_data), len(valid_data)
        print("Epoch %d: " % epoch, end='')
        print("train %d/%d (%.2f%%) " % (tr, ntr, 100*tr/ntr), end='')
        print("valid %d/%d (%.2f%%) " % (te, nte, 100*te/nte))

#### Helper functions
def sigmoid(z):
    """vectorized sigmoid function"""
    z = 1/(1 + np.exp(-z))
    return z

def sigmoid_grad(z):
    """vectorized gradient of the sigmoid function"""
    z = np.exp(z)/(1 + np.exp(z))**2
    return z

def unit(j, n):
    """return n x 1 unit vector with 1.0 at index j and zeros elsewhere"""
    unit_vector = np.zeros((n,1))
    unit_vector[j] = 1.0
    return unit_vector

def rand_mat(rows, cols, debug):
    """return random matrix of size rows x cols; if debug make repeatable"""
    eps = 0.12 # random values are in -eps...eps
    if debug:
        # use math.sin instead of random numbers
        vals = np.array([eps * math.sin(x+1) for x in range(rows * cols)])
        return np.reshape(vals, (rows, cols))
    else:
        return 2 * eps * np.random.rand(rows, cols) - eps
