import sys, os, math
dir =  os.getcwd()


# Multilayer Perceptron Regression
# ====================================
import matplotlib.pyplot as plt
import numpy as np, pandas
import seaborn as sns
# fix random seed for reproducibility
np.random.seed(7)

monthDict = {"Jan":"01", "Feb":"02", "Mar":"03", "Apr":"04", "May":"05", "Jun":"06", "Jul":"07",
            "Aug":"08", "Sep":"09", "Oct":"10", "Nov":"11", "Dec":"12"}
dataframe = pandas.read_csv(os.path.join(dir,'data','VolveData.csv'), header=0,
            usecols=['DATEPRD', 'NPD_WELL_BORE_CODE','ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
                     'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P',
                      'DP_CHOKE_SIZE', 'BORE_OIL_VOL', 'FLOW_KIND', 'BORE_GAS_VOL', 'BORE_WAT_VOL','BORE_WI_VOL'], engine='python')



# dfi = dataframe[dataframe['FLOW_KIND'] == 'injection']
wellCode = np.unique(dataframe['NPD_WELL_BORE_CODE'])


vars_list = ['BORE_OIL_VOL','ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
             'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHT_P',
             'AVG_WHP_P', 'DP_CHOKE_SIZE', 'BORE_GAS_VOL', 'BORE_WAT_VOL','BORE_WI_VOL', 'DATEPRD']


dataframe = dataframe.replace({',': ''}, regex=True)

dataframe = dataframe[dataframe['ON_STREAM_HRS']>0]

def outliers_z_score(vars):
  threshold = 3
  vars = vars.astype(np.float32)
  mean = np.mean(vars)
  std = np.std(vars)
  z_scores = [(var-mean) / std for var in vars]

  outliers = np.abs(z_scores)>threshold
  rest = np.abs(z_scores)<=threshold

  return rest, outliers


from scipy.stats import kde
num_features = 8
xs, ys = np.empty((0, num_features)), np.empty((0,1))
y_label = vars_list[0]
x_label = vars_list[9]

print('Processing correlation b/w {} and {}'.format(y_label, x_label))


for id, well in enumerate(wellCode):
  if well != 7289 and well!=7405:
    continue
  print('Processing well code {}'.format(well))

  cur_well = dataframe[dataframe['NPD_WELL_BORE_CODE']==well]
  cur_well = cur_well[cur_well['FLOW_KIND']=='production']

  X = np.array( [], dtype = np.float32)

  for x_label in vars_list[:10]:
    if x_label == 'BORE_OIL_VOL' or x_label =='AVG_ANNULUS_PRESS':
      continue
    cur_well = cur_well[cur_well[x_label]>0]

  raw_features = []
  for x_label in vars_list[:10]:
    if x_label == 'AVG_ANNULUS_PRESS':
      continue
    raw_features.append(cur_well[x_label].values )

  raw_features = np.array(raw_features, dtype=np.float32)
  raw_features = raw_features.T

  for id in range(raw_features.shape[1]):
    if id==0:
      continue
    out, _ = outliers_z_score(raw_features[:,id])
    raw_features = raw_features[out,:]


  if cur_well.empty:
    continue

  raw_features = (raw_features-np.mean(raw_features, axis=0) )/ np.std(raw_features, axis=0)
  bore_oil = raw_features[:,0]


  # NaN data cleaning for var_list[9] 'DP_CHOKE_SIZE'
  # where NaN data are replaced by neighbor mean values
  nanData = np.where(np.isnan(raw_features))
  if nanData[0].size>0:
    print('Error! Terminating program ...')
    exit()
  features = raw_features[:,1:]
  ys = np.append(ys, bore_oil)
  xs = np.append(xs, features, axis=0)
  # xs = np.append(xs,features, axis=1)
  # ys = np.append(ys,bore_oil)

print('Processing final data ...')



# split into train and test sets
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split( xs, ys, test_size=0.1, random_state=42)


# train_size = int(len(dataset) * 0.8)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#
print(len(train_X), len(test_X))

# # create and fit Multilayer Perceptron model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, activation='relu', input_dim=num_features))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
res = model.fit(train_X, train_y, validation_split=0.1, epochs=600, verbose=0)
# demonstrate prediction

yhat = model.predict(test_X, verbose=0)
# print(np.mean(np.abs(yhat-test_y)/np.abs(test_y)))


# # Estimate model performance
trainScore = model.evaluate(train_X, train_y, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(test_X, test_y, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


# summarize history for loss
plt.plot(res.history['loss'])
plt.plot(res.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=3):
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-look_back-1):
# 		a = dataset[i:(i+look_back), 0]
# 		dataX.append(a)
# 		dataY.append(dataset[i + look_back, 0])
# 	return np.array(dataX), np.array(dataY)
#
#
# # reshape into X=t and Y=t+1
# look_back = 3
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)


#
# model = Sequential()
# if look_back==1:
#   model.add(Dense(8, input_dim=look_back, activation='relu'))
#   model.add(Dense(1))
#   epochs =200
# else:
#   model.add(Dense(16, input_dim=look_back, activation='relu'))
#   model.add(Dense(8, activation='relu'))
#   model.add(Dense(1))
#   epochs = 800
#
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=epochs, batch_size=32, verbose=2)
#
# # Estimate model performance
# trainScore = model.evaluate(trainX, trainY, verbose=0)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
# testScore = model.evaluate(testX, testY, verbose=0)
# print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
#
#
