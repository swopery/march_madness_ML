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
        biases and weights are lists of length sizes-1.
        biases[i] is a column vector for layer i+1.
        weights[i] is a matrix for layers [i] and [i+1].
        """
        self.sizes = sizes
        self.debug = debug

        # use rand_mat(r, c, debug) to initialize these with
        # the right sizes r x c
        self.biases =  []
        self.weights = []
        for i in range(len(sizes)-1):
            self.weights.append(rand_mat(sizes[i+1], sizes[i], debug))
            self.biases.append(rand_mat(sizes[i+1], 1, debug))

    def feedforward(self, a):
        """Return the output of the network if a is input"""

        # from one layer to next, compute z = w*a+b, a = g(z)
        for i in range(1,len(self.sizes)):
            b = self.biases[i-1]
            w = self.weights[i-1]
            z = np.dot(w,a) + b
            a = sigmoid(z)
        return a

    def train(self, train_data, valid_data, epochs, mini_batch_size, alpha):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The ``train_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``valid_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
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
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        nL = len(self.sizes)
        m = len(mini_batch)

        # call self.backprop(x, y) and call the results
        # delta_b, delta_w
        # then add each of their elements to grad_b, grad_w

        # once you have the sums over the mini batch,
        # adjust the biases and weights by
        # subtracting (alpha/m) times the gradients
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            grad_b = [sum(x) for x in zip(delta_b, grad_b)]
            grad_w = [sum(x) for x in zip(delta_w, grad_w)]

        self.biases = [(bias - ((alpha/m) * gradient)) for bias, gradient in zip(self.biases, grad_b)]
        self.weights = [(weight - ((alpha/m) * gradient)) for weight, gradient in zip(self.weights, grad_w)]

    def backprop(self, x, y):
        """
        Return (grad_b, grad_w) representing the gradient of the cost
        function for a single training example (x, y).  grad_b and
        grad_w are layer-by-layer lists of numpy arrays, similar
        to self.biases and self.weights.
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        n = len(self.sizes) # number of layers

        # forward pass

        # same as feedforward, but store all a's and z's in lists
        a = [0] * n
        z = [0] * n
        a[0] = x # initial activation (z[0] is not used)
        for i in range(1, n): # 1 .. n-1
            b = self.biases[i-1]
            w = self.weights[i-1]
            z[i] = np.dot(w, a[i-1]) + b
            a[i] = sigmoid(z[i])

        # backward pass
        delta = [0] * n
        i = n-1 # index of last layer
        delta[i] = (a[i]-y) * sigmoid_grad(z[i])
        for i in range(n-2, 0, -1): # n-2 .. 1
            delta[i] = np.dot(np.transpose(self.weights[i]), delta[i+1])*sigmoid_grad(z[i])

        # compute gradients
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for i in range(0, n-1):
           grad_b[i] = delta[i+1]
           grad_w[i] = np.dot(delta[i+1],np.transpose(a[i]))
        return (grad_b, grad_w)

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
