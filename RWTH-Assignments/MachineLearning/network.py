import numpy as np
import random
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle


class Dataset:
    def __init__(self):
        self.index = 0

        self.obs = []
        self.classes = []
        self.num_obs = 0
        self.num_classes = 0
        self.indices = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_obs:
            self.index = 0
            raise StopIteration
        else:
            self.index += 1
            return self.obs[self.index - 1], self.classes[self.index - 1]

    def reset(self):
        self.index = 0

    def get_obs_with_target(self, k):
        index_list = [index for index, value in enumerate(self.classes) if value == k]
        return [self.obs[i] for i in index_list]

    def get_all_obs_class(self, shuffle=False):
        if shuffle:
            random.shuffle(self.indices)
        return [(self.obs[i], self.classes[i]) for i in self.indices]

    def get_mini_batches(self, batch_size, shuffle=False):
        if shuffle:
            random.shuffle(self.indices)

        batches = [(self.obs[self.indices[n:n + batch_size]],
                    self.classes[self.indices[n:n + batch_size]])
                   for n in range(0, self.num_obs, batch_size)]
        return batches


class IrisDataset(Dataset):
    def __init__(self, path):
        super(IrisDataset, self).__init__()
        self.file_path = path
        self.loadFile()
        self.indices = np.arange(self.num_obs)

    def loadFile(self):
        # load a comma-delimited text file into an np matrix
        resultList = []
        f = open(self.file_path, 'r')
        for line in f:
            line = line.rstrip('\n')  # "1.0,2.0,3.0"
            sVals = line.split(',')  # ["1.0", "2.0, "3.0"]
            fVals = list(map(np.float32, sVals))  # [1.0, 2.0, 3.0]
            resultList.append(fVals)  # [[1.0, 2.0, 3.0] , [4.0, 5.0, 6.0]]
        f.close()
        data = np.asarray(resultList, dtype=np.float32)  # not necessary
        self.obs = data[:, 0:4]
        self.classes = data[:, 4:7]
        self.num_obs = data.shape[0]
        self.num_classes = 3


# Activations
def tanh(x, deriv=False):
    """
    Implementation of the tanges hyperbolicus function. Derivative expects the x to be tanh(x).
    :param x: (array_like)
    :param deriv: (bool)
    :return:
    """
    if deriv:
        return 1.0 - x ** 2
    else:
        return np.tanh(x)


def sigmoid(x, deriv=False):
    """
    Implementation of the sigmoid function. Derivative expects the x to be sigmoid(x).
    :param x: (array_like)
    :param deriv: (bool)
    :return:
    """
    if deriv:
        return x * (1.0 - x)
    else:
        return 1 / (1 + np.exp(-x))


def linear(x, deriv=False):
    """
    Implementation of linear activation function
    :param x: (array like)
    :param deriv: (bool)
    :return:
    """
    if deriv:
        return x
    else:
        return x


def softmax(x):
    """
    Implementation of softmax function.
    :param x: (array like)
    :return:
    """
    exp_val = np.exp(x)
    return exp_val / np.sum(exp_val, axis=0)


class Layer:
    def __init__(self, num_input, num_output, activation=sigmoid):
        """
        :param num_input:  (int)
        :param num_output: (int)
        :param activation: (function)
        """
        print('Create layer with: {}x{} @ {}'.format(num_input, num_output, activation))
        self.ni = num_input
        self.no = num_output
        self.weights = np.zeros(shape=[self.ni, self.no], dtype=np.float32)
        self.biases = np.zeros(shape=[self.no], dtype=np.float32)
        self.initializeWeights()

        self.activation = activation

    def initializeWeights(self, method='normal'):
        """
        Initialize the weights of the weight matrix with one of several methods. Options are [normal, unity, zero]
        :param method: (string)
        :return:
        """
        if method == 'normal':
            weight = np.sqrt(2 / (self.no * self.ni))
            self.weights = weight * np.random.randn(self.ni, self.no)
        elif method == 'unity':
            self.weights = np.eye(self.ni, self.no)
        elif method == 'zero':
            self.weights = np.zeros(shape=[self.ni, self.no], dtype=np.float32)
        else:
            raise ValueError('method={} is not supported by initializeWeights'.format(method))

    def inference(self, x):
        """
        Computes the output of the Layer given an input vector x.
        :param x: (array_like)
        :return: (array_like)
        """
        t = np.dot(self.weights.T, x)
        t += self.biases
        y = self.activation(t)

        self.last_input = x
        self.last_output = y

        return y

    def backprop(self, error):
        """
        Calculates the residual error as well as the gradients based on an error signal.
        :param error: (array_like)
        :return: residual_error, gradients_w, gradients_b
        """

        ## backprop error
        t = [0] * self.no  # placeholder
        residual_error = [0] * self.ni  # error signal for previous layer
        for k in range(self.no):
            derivate = self.activation(self.last_output[k], deriv=True)
            t[k] = derivate * error[k]
        for j in range(self.ni):
            sum = 0.0
            for k in range(self.no):
                sum += t[k] * self.weights[j, k]
            residual_error[j] = sum

        ## Gradients
        # weights
        gradients_w = 0 * self.weights
        for j in range(self.ni):
            for k in range(self.no):
                gradients_w[j, k] = t[k] * self.last_input[j]

        # bias
        gradients_b = 0 * self.biases
        for k in range(self.no):
            gradients_b[k] = t[k]

        return residual_error, gradients_w, gradients_b


class BasicNeuralNetwork():
    def __init__(self, layer_sizes=[5], num_input=1, num_output=1, num_epoch=50, learning_rate=0.001,
                 mini_batch_size=None):
        self.layers = []
        self.ls = layer_sizes
        self.ni = num_input
        self.no = num_output
        self.lr = learning_rate
        self.num_epoch = num_epoch
        self.mbs = mini_batch_size

    def forward(self, x):
        """
        Compute the output of the neural network by continually evaluating each layer after the other given an input x. Afterwards a softmax is applied.
        :param x: (array_like)
        :return:
        """
        ## Forward pass
        signals = [x]
        for l in self.layers:
            s = l.inference(signals[-1])
            signals.append(s)
        return softmax(signals[-1])

    def train(self, train_dataset, eval_dataset=None, monitor_ce_train=True, monitor_accuracy_train=True,
              monitor_ce_eval=True, monitor_accuracy_eval=True, monitor_plot='monitor.png'):
        ce_train_array = []
        ce_eval_array = []
        acc_train_array = []
        acc_eval_array = []
        for e in range(self.num_epoch):
            if self.mbs:
                self.mini_batch_SGD(train_dataset)
            else:
                self.online_SGD(train_dataset)
            print('Finished training epoch: {}'.format(e))
            if monitor_ce_train:
                ce_train = self.ce(train_dataset)
                ce_train_array.append(ce_train)
                print('CE (train): {}'.format(ce_train))
            if monitor_accuracy_train:
                acc_train = self.accuracy(train_dataset)
                acc_train_array.append(acc_train)
                print('Accuracy (train): {}'.format(acc_train))
            if monitor_ce_eval:
                ce_eval = self.ce(eval_dataset)
                ce_eval_array.append(ce_eval)
                print('CE (eval): {}'.format(ce_eval))
            if monitor_accuracy_eval:
                acc_eval = self.accuracy(eval_dataset)
                acc_eval_array.append(acc_eval)
                print('Accuracy (eval): {}'.format(acc_eval))

        if monitor_plot:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
            line1, = ax[0].plot(ce_train_array, '--', linewidth=2, label='ce_train')
            line2, = ax[0].plot(ce_eval_array, label='ce_eval')

            line3, = ax[1].plot(acc_train_array, '--', linewidth=2, label='acc_train')
            line4, = ax[1].plot(acc_eval_array, label='acc_eval')

            ax[0].legend(loc='upper right')
            ax[1].legend(loc='upper left')
            ax[1].set_ylim([0, 1])

            plt.savefig(monitor_plot)

    def online_SGD(self, dataset):
        """
        Train the neural network in an online fashion. This means updating the weight parameters after each observation.
        The parameters are updated using basic stochastic gradient descent.
        :param dataset: (Dataset)
        :return:
        """
        for x, target in dataset:
            ## Forward pass
            signal = x
            for l in self.layers:
                signal = l.inference(signal)

            ## Backward pass
            # partial derivative of cross entropy error function and softmax results in:
            e = signal - target
            for l in reversed(self.layers):
                rs, gw, gb = l.backprop(e)
                l.weights -= self.lr * gw
                l.biases -= self.lr * gb

    def mini_batch_SGD(self, dataset):
        """
        In contrast to online_SDG the mini_bach variant aggregates several observations and updates the weights based on the average of the gradients.
        :param dataset: (Dataset)
        :return:
        """
        mini_batches = dataset.get_mini_batches(batch_size=self.mbs, shuffle=True)
        for batch in mini_batches:
            input_data = batch[0]
            target_data = batch[1]

            gradients = {}
            for idx in range(len(self.layers)):
                gradients[idx] = []  # make list to aggregate gradients of each layer

            for idx in range(len(input_data)):
                x = input_data[idx]
                target = target_data[idx]

                ## Forward pass
                signals = [x]
                for l in self.layers:
                    s = l.inference(signals[-1])
                    signals.append(s)

                ## Backward pass
                e = signals[-1] - target
                errors = [e]
                for idx in range(len(self.layers) - 1, -1, -1):  # go backwards through the layers
                    l = self.layers[idx]
                    rs, gw, gb = l.backprop(errors[-1])
                    errors.append(rs)
                    gradients[idx].append((gw, gb))

            # calculate update of weights based on gradients
            for idx in range(len(self.layers)):
                l = self.layers[idx]

                weight_update = 0 * l.weights
                bias_update = 0 * l.biases

                for gw, gb in gradients[idx]:
                    weight_update += gw
                    bias_update += gb

                weight_update *= 1 / len(gradients[idx])
                bias_update *= 1 / len(gradients[idx])

                l.weights -= self.lr * weight_update
                l.biases -= self.lr * bias_update

    def constructNetwork(self, hidden_activation='sigmoid', final_activation='sigmoid'):
        """
        Construct the Layers of the neural network based on the dimension of the input features (self.ni),
        the dimension of the hidden layers (self.ls), and the dimension of the classes (self.no). For the hidden and output
        layers different activation functions can be applied. If no activation function should be used, choose the 'linear'
        function for activation.
        Softmax is always applied during inference and does not need to be chosen as the activation for the final layer.
        :param hidden_activation: (str)
        :param final_activation:  (str)
        :return:
        """
        ci = self.ni
        for l in self.ls:
            self.layers.append(Layer(ci, l, activation=eval(hidden_activation)))
            ci = l
        self.layers.append(Layer(ci, self.no, activation=eval(final_activation)))

    def ce(self, dataset):
        """
        Compute the cross entropy score on a given dataset.
        :param dataset:
        :return:
        """
        ce = 0
        for x, true_label in dataset:
            t_hat = self.forward(x)
            ce -= np.sum(np.nan_to_num(true_label * np.log(t_hat) + (1 - true_label) * np.log(1 - t_hat)))
        return ce / dataset.num_obs

    def accuracy(self, dataset):
        cm = np.zeros(shape=[dataset.num_classes, dataset.num_classes], dtype=np.int)
        for x, t in dataset:
            t_hat = self.forward(x)
            c_hat = np.argmax(t_hat)  # index of largest output value
            c = np.argmax(t)
            cm[c, c_hat] += 1

        correct = np.trace(cm)
        return correct / dataset.num_obs

    def load(self, path=None):
        if not path:
            path = './network.save'
        with open(path, 'rb') as f:
            self.layers = pickle.load(f)

    def save(self, path=None):
        if not path:
            path = './network.save'
        with open(path, 'wb') as f:
            pickle.dump(self.layers, f)
