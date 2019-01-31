import sys, os, math
dir =  os.getcwd()


# Multilayer Perceptron Regression
# ====================================
import matplotlib.pyplot as plt
import numpy, pandas
from io import StringIO

# from keras.models import Sequential
# from keras.layers import Dense

# fix random seed for reproducibility
numpy.random.seed(7)

# load the datasetd
# dataframe = pandas.read_csv(os.path.join(dir,'data','Volve production data.csv'), usecols=[1], engine='python')
# dataframe = pandas.read_csv(StringIO(os.path.join(dir,'data','Volve production data.csv')),header=0, index_col=['DATEPRD'],
#             usecols=['WELL_BORE_CODE', 'ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
#                      'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P',
#                       'DP_CHOKE_SIZE', 'BORE_OIL_VOL', 'FLOW_KIND'], names=['WELL_BORE_CODE', 'ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
#                      'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P',
#                       'DP_CHOKE_SIZE', 'BORE_OIL_VOL', 'FLOW_KIND', 'DATEPRD'], parse_dates=['DATEPRD'], engine='c')


dataframe = pandas.read_csv(os.path.join(dir,'data','VolveData.csv'), header=0,
            usecols=['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
                     'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P',
                      'DP_CHOKE_SIZE', 'BORE_OIL_VOL', 'FLOW_KIND'], engine='python')


df = dataframe[dataframe['FLOW_KIND'] == 'production']
df = df.drop(['FLOW_KIND'],axis=1)
df = df.fillna(0)
df = df.replace({',': ''}, regex=True)
df_y = df['BORE_OIL_VOL']
y = df_y.values
df_X = df.drop(['BORE_OIL_VOL'], axis=1)
X = df_X.values

import numpy as np
dataset = np.concatenate([np.asarray(X),np.asarray([y]).T], axis=1)
dataset = dataset.astype('float32')


# # split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# # convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=1):
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-look_back-1):
# 		a = dataset[i:(i+look_back), 0]
# 		dataX.append(a)
# 		dataY.append(dataset[i + look_back, 0])
# 	return numpy.array(dataX), numpy.array(dataY)


# # reshape into X=t and Y=t+1
# look_back = 3
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)