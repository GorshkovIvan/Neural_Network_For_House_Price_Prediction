from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error


class Network(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, input_size, hiddenLayer1_size, output_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hiddenLayer1_size)
        self.fc2 = nn.Linear(hiddenLayer1_size, output_size)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Regressor():

    def __init__(self, x, nb_epoch=30, learning_rate=0.001, batch_size=100, device="cpu"):
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
        self.hiddenLayer1_size = 1  # we set this ourselves

        # Our code
        self.model = None
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device

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
        lb = preprocessing.OneHotEncoder(handle_unknown='ignore')
        ocean_prox = x['ocean_proximity']
        ocean_prox = np.array(ocean_prox)
        dummy_ocean_prox = lb.fit_transform(ocean_prox.reshape(-1, 1)).toarray()
        x = x.drop(['ocean_proximity'], axis=1)

        column_names = x.columns.tolist()
        x = x.values  # returns a numpy array
        minmax_scaler = preprocessing.MinMaxScaler()
        # standard_scaler = preprocessing.Standardizer()
        x_scaled = minmax_scaler.fit_transform(x)
        x = pd.DataFrame(x_scaled, columns=column_names)

        for i, dummy in enumerate(np.unique(ocean_prox)):
            x[dummy] = dummy_ocean_prox[:, i]

        # default value of 0 is  NOT final - set to proper default value
        x.fillna(0)
        x_tensor = torch.from_numpy(np.array(x)).float()

        # Preprocess Y
        if not training:
            y = y.fillna(0)
            return np.array(x), np.array(y)

        return np.array(x)

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

        model = Network(self.input_size, self.hiddenLayer1_size, self.output_size).to(self.device)
        loss_function = nn.MSELoss()
        optimiser = optim.SGD(model.parameters(), lr=self.learning_rate)

        # His code
        X, Y = self._preprocessor(x, y, training=True)  # Do not forget

        # Our code
        for epoch in range(self.nb_epoch):
            train_loader = torch.utils.data.DataLoader((X, Y), batch_size=self.batch_size, shuffle=True)

            for (inputs, labels) in enumerate(train_loader):
                # Forward pass
                optimiser.zero_grad()
                output = model(inputs)

                # Calculate Loss
                loss = loss_function(output, labels)
                loss.backward()

                # Update parameters
                optimiser.step()
                print(loss.item())

        self.model = model

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

        test_loader = torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=True)

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                pred_values, _ = torch.max(outputs.data, 1)

        return pred_values

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

        test_loader = torch.utils.data.DataLoader((X, Y), batch_size=self.batch_size, shuffle=True)

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                pred_value, _ = torch.max(outputs.data, 1)

        return mean_squared_error(Y, pred_value)

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
