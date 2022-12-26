import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import Sigmoid, ReLU, Linear, Tanh, MSELoss, LeakyReLU, ReLU6
from torch.optim import Adam
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import itertools

class Regressor(nn.Module):

    def __init__(self, x, nb_epoch = 1000, batch_size = 100, hidden_layers = [], activations = [], learning_rate=1e-3):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        super(Regressor, self).__init__()
        self.lb = None
        self.scaler = None
        X, _ = self._preprocessor(x, training = True)

        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

        if hidden_layers:
            self.hidden_layers = hidden_layers
        else:
            self.hidden_layers = [15, 15]
        
        if activations:
            self.activations = activations
        else:
            self.activations = ["relu", "relu"]

        self.neurons = [self.input_size] + self.hidden_layers + [self.output_size]
        self.activations = self.activations + ["identity"]

        self.layers = self.generate_layers(self.neurons, self.activations, self.input_size)
        self.model = nn.Sequential(*self.layers)
        self.loss_fun = MSELoss(reduction="sum")
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def generate_layers(self, neurons, activations, input_size):
        layers = []
        SIGMOID = "sigmoid"
        RELU = "relu"
        LEAKY_RELU = "leaky-relu"
        RELU_SIX = "relu-six"
        TANH = "tanh"

        for i, n_in in enumerate(neurons[:-1]):
            n_out = neurons[i + 1]
            layers.append(Linear(n_in, n_out))
            # Add Activation Layers
            if activations[i] == SIGMOID:
                layers.append(Sigmoid())
            elif activations[i] == RELU:
                layers.append(ReLU())
            elif activations[i] == LEAKY_RELU:
                layers.append(LeakyReLU())
            elif activations[i] == RELU_SIX:
                layers.append(ReLU6())
            elif activations[i] == TANH:
                layers.append(Tanh())

        return layers

    def forward(self, x):
        return self.model(x)

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x = x.fillna(x.mean(numeric_only=True))

        if y is not None:
            y = y.fillna(y.mean(numeric_only=True))

        x = x.to_numpy()

        if not training:
            ocean_proximity_encoded = self.lb.transform(x[:, -1])
            x = np.concatenate((x[:, :-1], ocean_proximity_encoded), axis=1)
            x = self.scaler.transform(x)
        else:
            # One Hot Encoding
            lb = LabelBinarizer()
            ocean_proximity_encoded = lb.fit_transform(x[:, -1])
            x = np.concatenate((x[:, :-1], ocean_proximity_encoded), axis=1)
            self.lb = lb
            # Normalise
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x)
            self.scaler = scaler
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        return (torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.to_numpy(np.float32)) if y is not None else None)
    

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        dataset = TensorDataset(X, Y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.nb_epoch):
            for (x_batch, y_batch) in data_loader:
                y_pred = self.forward(x_batch)
                loss = self.loss_fun(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, _ = self._preprocessor(x, training = False) # Do not forget
        self.model.eval()
        with torch.no_grad():
            predicted_values = self.forward(X).detach().numpy()
        return predicted_values
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        Y_pred = self.predict(x)
        return mean_squared_error(Y, Y_pred, squared=False)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

def split_train_test(data, start, end):
    test = data.iloc[start:end]
    train = pd.concat([data.iloc[0:start], data.iloc[end:]], axis=0)
    return train, test

def cross_validation(data, fold_indices, hidden_layers, activations, nb_epoch=200, batch_size=100):
    output_label = "median_house_value"
    current_errors = []
    for i in range(len(fold_indices) - 1):
        train, test = split_train_test(data, fold_indices[i], fold_indices[i + 1])
        x_train, y_train = train.loc[:, train.columns != output_label], train.loc[:, [output_label]]
        x_test, y_test = train.loc[:, train.columns != output_label], train.loc[:, [output_label]]
        regressor = Regressor(x_train, nb_epoch = nb_epoch, batch_size = batch_size, hidden_layers = hidden_layers, activations = activations)
        regressor.fit(x_train, y_train)
        error = regressor.score(x_test, y_test)
        current_errors.append(error)
    return np.mean(current_errors)

def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    data = pd.read_csv("housing.csv")
    N = len(data.index)
    folds = 10
    hyper_parameters = {}

    # Parameters To Try
    hidden_layers = [1, 3, 5, 7, 10]
    neurons_in_layers = [5, 10, 15, 20]
    epoches = [250, 500, 1000, 1500, 2000]
    batch_sizes = [10, 50, 100, 150, 200]
    activations = ["relu", "leaky-relu", "relu-six"]

    lowest_error = float("inf")

    fold_indices = np.linspace(0, N + 1, folds + 1, dtype=int)

    # Find inital parameters
    for n_hidden_layers in hidden_layers:
        for neurons in neurons_in_layers:
            for activation in activations:
                activations = [activation for _ in range(n_hidden_layers)]
                hidden_layers = [neurons for _ in range(n_hidden_layers)]
                mean_error = cross_validation(data, fold_indices, hidden_layers, activations)
                if mean_error < lowest_error:
                    hyper_parameters["n_hidden_layers"] = n_hidden_layers
                    hyper_parameters["neurons"] = neurons
                    hyper_parameters["activation_fun"] = activation
                    lowest_error = mean_error
    
    # hyper_parameters = {'n_hidden_layers': 10, 'neurons': 20, 'activation_fun': 'leaky-relu'}
    print(f"Initial Hyperparameters: {hyper_parameters} | Error: {lowest_error}")

    x_values, y_values = [], []

    # Fix Hidden Layers & Neurons
    for activation in activations:
        activations = [activation for _ in range(hyper_parameters["n_hidden_layers"])]
        hidden_layers = [hyper_parameters["neurons"] for _ in range(hyper_parameters["n_hidden_layers"])]
        mean_error = cross_validation(data, fold_indices, hidden_layers, activations)
        x_values.append(activation)
        y_values.append(mean_error)
        if mean_error < lowest_error:
            hyper_parameters["activation_fun"] = activation
            lowest_error = mean_error

    plt.plot(x_values, y_values)
    plt.xlabel("Activation Function")
    plt.ylabel("Mean Error")
    plt.savefig('./graphs/activation_only.png')
    plt.clf()
    
    print(f"Changing Only Activation: {hyper_parameters} | Error: {lowest_error}")
    
    x_values, y_values = [], []

    # Fix Hidden Layers & Activations
    for neurons in neurons_in_layers:
        activations = [hyper_parameters["activation_fun"] for _ in range(hyper_parameters["n_hidden_layers"])]
        hidden_layers = [neurons for _ in range(hyper_parameters["n_hidden_layers"])]
        mean_error = cross_validation(data, fold_indices, hidden_layers, activations)
        x_values.append(neurons)
        y_values.append(mean_error)
        if mean_error < lowest_error:
            hyper_parameters["neurons"] = neurons
            lowest_error = mean_error
    
    plt.plot(x_values, y_values)
    plt.xlabel("Number of Neurons In Each Layer")
    plt.ylabel("Mean Error")
    plt.savefig('./graphs/neurons_only.png')
    plt.clf()

    print(f"Changing Neurons Only: {hyper_parameters} | Error: {lowest_error}")
        
    # Mixed Neurons
    for permutation in list(itertools.permutations(neurons_in_layers)):
        for neurons in permutation:
            activations = [hyper_parameters["activation_fun"] for _ in range(hyper_parameters["n_hidden_layers"])]
            hidden_layers = [neurons for _ in range(hyper_parameters["n_hidden_layers"])]
            mean_error = cross_validation(data, fold_indices, hidden_layers, activations)
            if mean_error < lowest_error:
                hyper_parameters["neurons"] = neurons
                lowest_error = mean_error
    
    print(f"Using Mixed Number Of Neurons: {hyper_parameters} | Error: {lowest_error}")

    x_values, y_values = [], []

    # Fix Neurons & Activations
    for n_hidden_layers in hidden_layers:
        activations = [hyper_parameters["activation_fun"] for _ in range(n_hidden_layers)]
        hidden_layers = [hyper_parameters["neurons"] for _ in range(n_hidden_layers)]
        mean_error = cross_validation(data, fold_indices, hidden_layers, activations)
        x_values.append(n_hidden_layers)
        y_values.append(mean_error)
        if mean_error < lowest_error:
            hyper_parameters["n_hidden_layers"] = n_hidden_layers
            lowest_error = mean_error

    plt.plot(x_values, y_values)
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Mean Error")
    plt.savefig('./graphs/hidden_layers_only.png')
    plt.clf()

    print(f"Changing Only Hidden Layers: {hyper_parameters} | Error: {lowest_error}")

    x_values, y_values = [], []

    # Modify Epoch
    for epoch in epoches:
        activations = [hyper_parameters["activation_fun"] for _ in range(hyper_parameters["n_hidden_layers"])]
        hidden_layers = [hyper_parameters["neurons"] for _ in range(hyper_parameters["n_hidden_layers"])]
        mean_error = cross_validation(data, fold_indices, hidden_layers, activations, nb_epoch=epoch)
        x_values.append(epoch)
        y_values.append(mean_error)
        if mean_error < lowest_error:
            hyper_parameters["epoch"] = epoch
            lowest_error = mean_error
    
    plt.plot(x_values, y_values)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Error")
    plt.savefig('./graphs/epoch_only.png')
    plt.clf()

    print(f"Changing Only Epoch: {hyper_parameters} | Error: {lowest_error}")

    x_values, y_values = [], []

    # Modify Batchsize
    for batch_size in batch_sizes:
        activations = [hyper_parameters["activation_fun"] for _ in range(hyper_parameters["n_hidden_layers"])]
        hidden_layers = [hyper_parameters["neurons"] for _ in range(hyper_parameters["n_hidden_layers"])]
        mean_error = cross_validation(data, fold_indices, hidden_layers, activations, batch_size=batch_size, nb_epoch=hyper_parameters["epoch"])
        x_values.append(batch_size)
        y_values.append(mean_error)
        if mean_error < lowest_error:
            hyper_parameters["batch_size"] = batch_size
            lowest_error = mean_error
    
    plt.plot(x_values, y_values)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Error")
    plt.savefig('./graphs/batch_size_only.png')
    plt.clf()

    print(f"Changing Only Batchsize: {hyper_parameters} | Error: {lowest_error}")

    return hyper_parameters
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

def train_with_hyper_parameters(hyper_parameters):
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv") 
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]
    activations = [hyper_parameters["activation_fun"] for _ in range(hyper_parameters["n_hidden_layers"])]
    hidden_layers = [hyper_parameters["neurons"] for _ in range(hyper_parameters["n_hidden_layers"])]
    regressor = Regressor(x_train, nb_epoch = 100, batch_size = 100, hidden_layers = hidden_layers, activations = activations)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))

def model_evaluation_graphs():
    output_label = "median_house_value"

    data = pd.read_csv("housing.csv") 

    end = int(len(data) * 0.30)

    training, test = split_train_test(data, 0, end)

    x_train = training.loc[:, data.columns != output_label]
    y_train = training.loc[:, [output_label]]

    x_test = test.loc[:, data.columns != output_label]
    y_test = test.loc[:, [output_label]]
    regressor = Regressor(x_train, nb_epoch = 100)
    regressor.fit(x_train, y_train)

    # regressor = load_regressor()

    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))

    # Plots Graph
    y_test = y_test.to_numpy(np.float32)
    y_pred = regressor.predict(x_test)

    plt.figure("Actual vs Predicted")
    plt.scatter(y_test, y_pred, c="crimson")
    plt.yscale("log")
    plt.xscale("log")
    p1 = max(np.max(y_pred), np.max(y_test))
    p2 = min(np.min(y_pred), np.min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.axis("equal")
    plt.show()

def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    # save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    # model_evaluation_graphs()
    example_main()
    # hyper_parameters = RegressorHyperParameterSearch()
    # print(hyper_parameters) # {'n_hidden_layers': 7, 'neurons': 15, 'activation_fun': 'relu'}
    # train_with_hyper_parameters(hyper_parameters)

