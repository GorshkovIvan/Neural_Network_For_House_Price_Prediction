from typing import Any

import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import pandas as pd
import sys
import random
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def include_dummies(x):
    dummy_variables = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

    lb = preprocessing.OneHotEncoder(handle_unknown='ignore')
    ocean_prox = x['ocean_proximity']
    ocean_prox = np.array(ocean_prox)
    dummy_ocean_prox = lb.fit_transform(ocean_prox.reshape(-1, 1)).toarray()
    x = x.drop(['ocean_proximity'], axis=1)

    for i, dummy in enumerate(np.unique(ocean_prox)):
        x[dummy] = dummy_ocean_prox[:, i]

    for name in dummy_variables:
        if name not in x:
            x[name] = np.zeros(len(x))

    x = x.sort_index(axis=1)

    return x


class Network(nn.Module):

    def __init__(self, input_size, hiddenLayer1_size, hiddenLayer2_size, output_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hiddenLayer1_size)
        self.fc2 = nn.Linear(hiddenLayer1_size, hiddenLayer2_size)
        self.fc3 = nn.Linear(hiddenLayer2_size, output_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


class Regressor():

    def __init__(self, x, nb_epoch=30, learning_rate=0.002, batch_size=100, layer1_neurons=50,
                 layer2_neurons=50):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=True)
        self.x = x

        self.input_size = X.shape[1]
        self.output_size = 1
        self.hiddenLayer1_size = layer1_neurons  # we set this ourselves
        self.hiddenLayer2_size = layer2_neurons  # we set this ourselves

        self.model = None
        self.prev_model = None
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.scaler = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
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
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # turn the categorical variables into dummies using One-Hot Encoding
        x = include_dummies(x)

        # Scaling the data using Min Max
        column_names = x.columns.tolist()
        x = x.values

        # new preprocessing values required if training
        if training:
            self.scaler = preprocessing.MinMaxScaler()
            self.scaler = self.scaler.fit(x)

        x = self.scaler.transform(x)
        x = pd.DataFrame(x, columns=column_names)

        # fill in NaN x values with a random uniform distribution U~[0,1]
        x = x.fillna(random.uniform(0, 1))
        x_tensor = torch.from_numpy(np.array(x)).float()

        # fill in NaN y values with a random uniform distribution U~[0,1]
        if y is not None:
            y = y.fillna(random.uniform(0, 1))
            y_tensor = torch.from_numpy(y.to_numpy()).float()

        return x_tensor, (y_tensor if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

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

        self.model = Network(self.input_size, self.hiddenLayer1_size, self.hiddenLayer2_size, self.output_size).to(
            "cpu")
        loss_function = nn.MSELoss()
        optimiser = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1 / 9, random_state=4, shuffle=True)
        X, Y = self._preprocessor(x_train, y_train, training=True)  # Do not forget

        # Split X, Y into x_train, x_val and y_train, y_val
        dataset = torch.utils.data.TensorDataset(X, Y)

        loss_list = []
        score_list = []
        previous_score = -sys.maxsize
        for epoch in range(self.nb_epoch):
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            running_loss = 0.0
            # Calculate validation mse score
            current_score = self.score(x_val, y_val)
            score_list.append(current_score)
            if current_score < previous_score:
                loss_list.append(loss_list[-1])
                self.model = self.prev_model
                break

            previous_score = current_score
            self.prev_model = self.model
            for i, (inputs, labels) in enumerate(train_loader, 0):
                # Forward pass
                optimiser.zero_grad()
                output = self.model(inputs)

                # Calculate Loss
                loss = loss_function(output, labels)
                loss.backward()

                # Update parameters
                optimiser.step()

                running_loss += loss.item()

            print("Epoch [{}/{}], Average Training Loss: {}, Validation Loss: {}"
                  .format(epoch + 1, self.nb_epoch, running_loss / len(train_loader), current_score))
            loss_list.append(running_loss / len(train_loader))

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
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, _ = self._preprocessor(x, training=False)  # Do not forget

        predictions = []
        with torch.no_grad():
            for i, value in enumerate(X):
                outputs = self.model(value)
                predictions = np.append(predictions, outputs)

        return predictions

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
        X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget

        predictions = []
        with torch.no_grad():
            for i, value in enumerate(X):
                outputs = self.model(value)
                predictions = np.append(predictions, outputs)

        return -mean_squared_error(Y, predictions)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def get_params(self, deep=True):
        return {"x": self.x, "nb_epoch": self.nb_epoch, "learning_rate": self.learning_rate,
                "batch_size": self.batch_size, "layer1_neurons": self.hiddenLayer1_size,
                "layer2_neurons": self.hiddenLayer2_size}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self


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


def RegressorHyperParameterSearch(model, x_train, y_train, x_test, y_test):
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

    param_grid = {'x': [x_train],
                  'nb_epoch': [50],
                  'learning_rate': [0.1, 0.002, 0.0005],
                  'batch_size': [10, 50, 100],
                  'layer1_neurons': [5, 25, 50],
                  'layer2_neurons': [5, 25, 50]}

    grid = sklearn.model_selection.GridSearchCV(model, param_grid, refit=True, cv=4, n_jobs=-1)

    # fitting the model for grid search
    grid_result = grid.fit(x_train, y_train)

    return grid_result.best_params_

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Splitting dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=3, shuffle=True)

    # Training the regressor only on the training dataset
    regressor = Regressor(x_train, nb_epoch=10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Evaluation
    # Printing out the MSE of our regressor evaluated using y_test
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

