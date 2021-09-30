import tgan
from tgan.model import TGANModel
import pandas as pd
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
# data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
#  columns=  + ['target'])

data1 = pd.DataFrame(iris['data'])
continuous_columns_iris = data1.columns
data_dum = pd.get_dummies(data1)
data_dum = pd.DataFrame(data_dum)
tgan = TGANModel(continuous_columns_iris, batch_size=10)
tgan.fit(data_dum)
# num_samples = 100
# samples = tgan.sample(num_samples)
# print(samples)
