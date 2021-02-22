import torch
import pickle
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing

output_label = "median_house_value"

# Use pandas to read CSV data as it contains various object types
# Feel free to use another CSV reader tool
# But remember that LabTS tests take Pandas Dataframe as inputs
data = pd.read_csv("housing.csv")

# Our code
print(data.head())
ocean_prox = data['ocean_proximity']
ocean_prox = np.array(ocean_prox)
print("Ocean prox:")
print(ocean_prox)

lb = preprocessing.OneHotEncoder(handle_unknown='ignore')
#ocean_prox = np.reshape(ocean_prox, (-1, 1))
#dummy_ocean_prox = lb.fit_transform(ocean_prox)
#print("Dummy Ocean Prox:")
#print(dummy_ocean_prox)

print(np.unique(ocean_prox))
dummy_ocean_prox = lb.fit_transform(ocean_prox.reshape(-1, 1)).toarray()
print("Dummy Ocean Prox:")
print(dummy_ocean_prox)

#print(lb.transform(dummy_ocean_prox).toarray())

data = data.drop(['ocean_proximity'], axis=1)
print("Data head:")

column_names = data.columns.tolist()
x = data.values  # returns a numpy array
minmax_scaler = preprocessing.MinMaxScaler()
#standard_scaler = preprocessing.Standardizer()
x_scaled = minmax_scaler.fit_transform(x)
data = pd.DataFrame(x_scaled, columns = column_names)

for i, dummy in enumerate(np.unique(ocean_prox)):
    data[dummy] = dummy_ocean_prox[:,i]

print(data.head(10))