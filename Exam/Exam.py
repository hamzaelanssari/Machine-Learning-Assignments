import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
from tkinter.ttk import *
import tkinter as tk
from tkinter import *


class softmax:
    def function(x):
        exps = np.exp(x)
        return exps / np.sum(exps)

    def derivative(x):
        s = x.reshape((-1, 1))
        d_s = np.diagflat(x) - np.dot(s, s.T)
        return d_s


class tanh:
    def function(x):
        t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return t

    def derivative(x):
        return 1 - x ** 2


class relu:
    def function(x):
        return np.where(x > 0, x, 0)

    def derivative(x):
        return np.where(x > 0, 1, 0)


class sigmoid:
    def function(x):
        return 1 / (1 + np.exp(-x))

    def derivative(x):
        return x * (1 - x)


class sign:
    def function(x):
        return np.where(x < 0, -1, 1)

    def derivative(x):
        return 0


class step:
    def function(x):
        return np.where(x < 0, 0, 1)

    def derivative(x):
        return 0


class leaky_relu:
    def function(x):
        return np.where(x > 0, x, 0.01 * x)

    def derivative(x):
        return np.where(x > 0, 1, 0.01)


class MultiLayerNN:
    def __init__(self, epochs, lr, num_input_layers, num_hidden_layers, num_output_layers):
        self.losses = []

        # Epochs
        self.epochs = epochs

        # Learning rate
        self.lr = lr

        # Random weights and bias initialization
        self.hidden_weights = np.random.uniform(size=(num_input_layers, num_hidden_layers))
        self.hidden_bias = np.random.uniform(size=(1, num_hidden_layers))
        self.output_weights = np.random.uniform(size=(num_hidden_layers, num_output_layers))
        self.output_bias = np.random.uniform(size=(1, num_output_layers))

        # Activation function initialization
        self.hidden_function = None
        self.output_function = None

    def hidden_activation_function(self, hidden_activation_function):
        if hidden_activation_function == 'sigmoid':
            self.hidden_function = sigmoid
        elif hidden_activation_function == 'softmax':
            self.hidden_function = softmax
        elif hidden_activation_function == 'relu':
            self.hidden_function = relu
        elif hidden_activation_function == 'tanh':
            self.hidden_function = tanh
        else:
            self.hidden_function = relu

    def output_activation_function(self, output_activation_function):
        if output_activation_function == 'sigmoid':
            self.output_function = sigmoid
        elif output_activation_function == 'softmax':
            self.output_function = softmax
        elif output_activation_function == 'relu':
            self.output_function = relu
        elif output_activation_function == 'tanh':
            self.output_function = tanh
        else:
            self.output_function = softmax

    def activation_function(self, hidden_activation_function, output_activation_function):
        self.hidden_activation_function(hidden_activation_function)
        self.output_activation_function(output_activation_function)

    # Loss Function
    def loss(self, yp, y):
        return (1 / 2) * np.square(yp - y)

    def forward(self, inputs):
        # Hidden Layer
        hidden_layer_activation = np.dot(inputs, self.hidden_weights) + self.hidden_bias

        hidden_layer_output = self.hidden_function.function(hidden_layer_activation)

        # Output Layer
        output_layer_activation = np.dot(hidden_layer_output, self.output_weights) + self.output_bias
        predicted_output = self.output_function.function(output_layer_activation)
        return hidden_layer_output, predicted_output

    def backward(self, hidden_layer_output, predicted_output):
        # Output Layer
        error = self.expected_output - predicted_output
        d_predicted_output = error * self.output_function.derivative(predicted_output)

        # Hidden Layer
        error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
        d_hidden_layer = error_hidden_layer * self.hidden_function.derivative(hidden_layer_output)
        return d_hidden_layer, d_predicted_output

    def fit(self, X, y):
        np.random.seed(0)

        # Input data
        self.inputs = X
        self.expected_output = y.reshape(len(y), 1)

        # Training algorithm
        for _ in range(self.epochs):
            # Forward Propagation
            hidden_layer_output, predicted_output = self.forward(self.inputs)

            # Backpropagation
            d_hidden_layer, d_predicted_output = self.backward(hidden_layer_output, predicted_output)

            # Updating Weights and Biases
            self.output_weights += hidden_layer_output.T.dot(d_predicted_output) * self.lr
            self.output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * self.lr
            self.hidden_weights += self.inputs.T.dot(d_hidden_layer) * self.lr
            self.hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.lr

            # Loss
            loss_ = self.loss(self.expected_output, predicted_output)[0]
            self.losses.append(loss_)

    def predict(self, inputs):
        predicted_output = self.forward(inputs)[1]
        predicted_output = np.squeeze(predicted_output)
        return np.where(predicted_output >= 0.5, 1, 0)

    def info_of_classification(self):
        predicted_output = self.predict(self.inputs)
        print("Accuracy : ", self.accuracy(predicted_output, self.expected_output))
        print("F1 score : ", self.f1_score(predicted_output, self.expected_output))
        print("Recall score : ", self.recall_score(predicted_output, self.expected_output))
        print("Precision score: ", self.precision_score(predicted_output, self.expected_output))
        print("Confusion Matrix : ", self.confusion_matrix(predicted_output, self.expected_output))

    def accuracy(self, predicted_output, outputs):
        return accuracy_score(predicted_output, outputs)

    def confusion_matrix(self, predicted_output, outputs):
        return confusion_matrix(outputs, predicted_output)

    def f1_score(self, predicted_output, outputs):
        return f1_score(outputs, predicted_output)

    def recall_score(self, predicted_output, outputs):
        return recall_score(outputs, predicted_output)

    def precision_score(self, predicted_output, outputs):
        return precision_score(outputs, predicted_output)

    def draw_loss(self):
        plt.plot(self.losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


class index:
    def __init__(self, root):
        self.root = root
        self.root.configure()
        self.root.geometry('1300x750')
        self.root.resizable(False, False)
        self.header_inputs()
        self.draw_classification()
        self.draw_loss()
        self.infos()

    def labels(self, root):
        data = Label(root, text=" Select data source  ", bg='#BBBBBB',
                         font=("Goudy old style", 10)).place(x=50, y=40)
        epochs = Label(root, text="Epochs", bg='#BBBBBB',
                      font=("Goudy old style", 10)).place(x=50, y=70)

        lr = Label(root, text="Learning rate", bg='#BBBBBB',
                      font=("Goudy old style", 10)).place(x=50, y=100)

        input_layers = Label(root, text="Number of input layer neurons", bg='#BBBBBB',
                         font=("Goudy old style", 10)).place(x=50, y=130)
        hidden_layers = Label(root, text="Number of hidden layer neurons", bg='#BBBBBB',
                         font=("Goudy old style", 10)).place(x=50, y=160)
        output_layers = Label(root, text="Number of output layer neurons", bg='#BBBBBB',
                         font=("Goudy old style", 10)).place(x=50, y=190)
        hidden_function = Label(root, text="Hidden activation function", bg='#BBBBBB',
                         font=("Goudy old style", 10)).place(x=50, y=220)
        output_function = Label(root, text="Output activation function", bg='#BBBBBB',
                         font=("Goudy old style", 10)).place(x=50, y=250)


    def header_inputs(self):
        header = Frame(self.root, bg='#BBBBBB')
        header.place(x=10, y=10, height=580, width=500)

        header_inputs = Frame(header, bg='#03597C')
        header_inputs.place(x=0, y=0, height=30, width=500)
        txt = f'Configurez les inputs'
        conf = Label(header_inputs, text=txt, font=("Anaheim", 10),
                        bg="#03597C",
                        fg="white").place(x=10, y=5)
        conf_btn = Button(header, text="Classifier",
                                  fg="white", bd=0,
                                  bg="#03597C",
                                  font=("times new roman", 11)).place(x=385, y=540, width=100, height=30)
        self.labels(header)



    def draw_classification(self):
        classification = Frame(self.root, bg='#BBBBBB')
        classification.place(x=550, y=10, height=355, width=740)

        classification_inputs = Frame(classification, bg='#03597C')
        classification_inputs.place(x=0, y=0, height=30, width=740)
        txt = f'Configurez les inputs'
        conf = Label(classification_inputs, text=txt, font=("Anaheim", 10),
                        bg="#03597C",
                        fg="white").place(x=10, y=5)

    def draw_loss(self):
        loss = Frame(self.root, bg='#BBBBBB')
        loss.place(x=550, y=385, height=355, width=740)

        loss_inputs = Frame(loss, bg='#03597C')
        loss_inputs.place(x=0, y=0, height=30, width=740)
        txt = f'Configurez les inputs'
        conf = Label(loss_inputs, text=txt, font=("Anaheim", 10),
                        bg="#03597C",
                        fg="white").place(x=10, y=5)

    def infos(self):
        infos = Frame(self.root, bg='#BBBBBB')
        infos.place(x=10, y=610, height=130, width=500)

        infos_inputs = Frame(infos, bg='#03597C')
        infos_inputs.place(x=0, y=0, height=30, width=500)
        txt = f'Configurez les inputs'
        conf = Label(infos_inputs, text=txt, font=("Anaheim", 10),
                        bg="#03597C",
                        fg="white").place(x=10, y=5)
        lbl_design = Label(infos, bg="#2186A4").place(x=250, y=30, width=1, height=100)


if __name__ == '__main__':
    root = Tk(className=' XOR Problem')
    obj = index(root)
    root.mainloop()
'''    rng = np.random.RandomState(0)
    X = rng.randn(300, 2)
    y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0), dtype=int)
    epochs = 10000
    lr = 0.01
    inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 4, 1
    activation_functions = ['sigmoid', 'softmax', 'relu', 'tanh', 'leaky_relu', 'sign', 'step']
    model = MultiLayerNN(epochs, lr, inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons)
    model.activation_function(activation_functions[2], activation_functions[0])
    model.fit(X, y)
    model.info_of_classification()
    fig = plt.figure(figsize=(10, 8))
    fig = plot_decision_regions(X=X, y=y, clf=model, legend=2)
    plt.title("XOR MLNN from scratch")
    plt.show()
    model.draw_loss()'''
