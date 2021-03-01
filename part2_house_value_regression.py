from typing import Any

import matplotlib.pyplot as plt
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, classification_report
from numpy.random import default_rng
from sklearn.model_selection import train_test_split


class Network(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        print("this shouldn't be called")

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

    def __init__(self, x, nb_epoch=30, learning_rate=0.01, batch_size=100, device="cpu"):
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

        # Replace this code with your own
        X = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.hiddenLayer1_size = 200  # we set this ourselves
        self.hiddenLayer2_size = 200 # we set this ourselves

        # Our code
        self.model = None
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device

        # To save scaler
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

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        # our code

        # Preprocess x
        # Encoding textual data using One Hot
        lb = preprocessing.OneHotEncoder(handle_unknown='ignore')
        ocean_prox = x['ocean_proximity']
        ocean_prox = np.array(ocean_prox)
        dummy_ocean_prox = lb.fit_transform(ocean_prox.reshape(-1, 1)).toarray()
        x = x.drop(['ocean_proximity'], axis=1)

        # Scaling the data using Min Max
        column_names = x.columns.tolist()
        x = x.values  # returns a numpy array

        if training:  # new preprocessing values required if training
            self.scaler = preprocessing.MinMaxScaler()  # set scaler
            self.scaler.fit(x) # fit scaler to x and save for later

        x = self.scaler.transform(x) # transform x using scaler
        x = pd.DataFrame(x, columns=column_names) # turn x into dataframe

        for i, dummy in enumerate(np.unique(ocean_prox)):
            x[dummy] = dummy_ocean_prox[:, i]

        # default value of 0 is  NOT final - set to proper default value
        x = x.fillna(0)
        x_tensor = torch.from_numpy(np.array(x)).float()

        # Preprocess Y

        if y is not None:
            y = y.fillna(0)
            y_tensor = torch.from_numpy(np.array(y)).float()
            return x_tensor, y_tensor

        return x_tensor


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

        self.model = Network(self.input_size, self.hiddenLayer1_size, self.hiddenLayer2_size, self.output_size).to(self.device)
        loss_function = nn.MSELoss()
        #optimiser = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        optimiser = optim.Adam(self.model.parameters())
        print(self.learning_rate)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

        # His code
        X, Y = self._preprocessor(x_train, y_train, training=True)  # Do not forget

        # Split X, Y into x_train, x_val and y_train, y_val


        dataset = torch.utils.data.TensorDataset(X, Y)

        # Our code
        loss_list = []
        previous_score = sys.maxsize
        for epoch in range(self.nb_epoch):
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            running_loss = 0.0

            # Calculate validation mse score
            current_score = self.score(x_val, y_val)
            if current_score > previous_score:
                break
            previous_score = current_score

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
                  .format(epoch + 1, self.nb_epoch, running_loss / (i + 1), current_score))
            loss_list.append(running_loss / (i + 1))

        plt.plot(range(len(loss_list)), loss_list)
        plt.ylabel("Average training loss per epoch")
        plt.xlabel("Epoch")
        plt.show()

        #self.save_regressor("trained_model.pickle")

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

        X = self._preprocessor(x, training=False)  # Do not forget

        predictions = []
        with torch.no_grad():
            for i, value in enumerate(X):
                outputs = self.model(value)
                predictions.append(outputs)
        return np.array(predictions)

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
                predictions.append(outputs)

        return mean_squared_error(Y, np.array(predictions))

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

    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'gamma': ['scale', 'auto'],
                  'kernel': ['linear']}

    grid = sklearn.model_selection.GridSearchCV(model, param_grid, refit=True, verbose=3, n_jobs=-1)

    # fitting the model for grid search
    grid.fit(x_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(x_test)

    # print classification report
    print(classification_report(y_test, grid_predictions))

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Our code
    random_generator = default_rng()
    shuffled_indices = random_generator.permutation(len(x_train))
    x_train = x_train[shuffled_indices[:]]
    y_train = y_train[shuffled_indices[:]]
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch=10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
