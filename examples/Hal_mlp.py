import sys, os, math
dir =  os.getcwd()


# Multilayer Perceptron Regression
# ====================================
import matplotlib.pyplot as plt
import numpy as np, pandas

# fix random seed for reproducibility
np.random.seed(7)


dataframe = pandas.read_csv(os.path.join(dir,'data','VolveData.csv'), header=0,
            usecols=['NPD_WELL_BORE_CODE','ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
                     'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P',
                      'DP_CHOKE_SIZE', 'BORE_OIL_VOL', 'FLOW_KIND', 'BORE_GAS_VOL', 'BORE_WAT_VOL','BORE_WI_VOL'], engine='python')



# dfi = dataframe[dataframe['FLOW_KIND'] == 'injection']
wellCode = np.unique(dataframe['NPD_WELL_BORE_CODE'])

goodwell = [5599, 7289, 7405]

vars_list = ['BORE_OIL_VOL','ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
             'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHT_P',
             'AVG_WHP_P', 'DP_CHOKE_SIZE', 'BORE_GAS_VOL', 'BORE_WAT_VOL','BORE_WI_VOL']

vars_list2 = ['BORE_OIL_VOL','ON_STREAM_HRS', 'AVG_CHOKE_SIZE_P', 'AVG_WHT_P',
             'AVG_WHP_P', 'DP_CHOKE_SIZE']


vars_list3 = ['BORE_OIL_VOL','ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
             'AVG_DP_TUBING',  'AVG_CHOKE_SIZE_P', 'AVG_WHT_P', 'AVG_WHP_P', 'DP_CHOKE_SIZE']


dataframe = dataframe.replace({',': ''}, regex=True)

dataframe = dataframe[dataframe['ON_STREAM_HRS']>0]

from scipy.stats import kde

xs, ys = np.zeros([]), np.zeros([])
y_label = vars_list[0]
x_label = vars_list[2]

print('Processing correlation b/w {} and {}'.format(y_label, x_label))
for id, well in enumerate(wellCode):
  print('Processing well code {}'.format(well))

  cur_well = dataframe[dataframe['NPD_WELL_BORE_CODE']==well]
  # if cur_well['FLOW_KIND'].values[0]=='injection':
  #   continue
  cur_well = cur_well[cur_well['FLOW_KIND']=='production']

  cur_well = cur_well[cur_well[x_label]>0]


  if cur_well.empty:
    continue
  y = np.array(cur_well[y_label].values, dtype=np.float16)
  x = np.array(cur_well[x_label].values, dtype=np.float16)
  xsh = np.shape(x)
  ysh = np.shape(y)


  # NaN data cleaning for var_list[9] 'DP_CHOKE_SIZE'
  # where NaN data are replaced by neighbor mean values
  nanData = np.where(np.isnan(x))
  rg = 2
  if nanData[0].size>0:
    exit()
    for i, idx in enumerate(nanData[0]):
      rg = 2
      while 1:
        ipt = np.sum(~np.isnan(x[ max(0,nanData[0][i]-rg): min(xsh[0], nanData[0][i]+rg) ]))
        if ipt<4:
          rg+=1
        else:
          window = x[max(0, nanData[0][i] - rg): min(xsh[0], nanData[0][i] + rg)]
          ipt_data = np.mean(window[~np.isnan(window)])
          break
        if(rg>10):
          break
      if(rg>10):
        break
      x[nanData[0][i]] = ipt_data

  if rg>10 or xsh[0]==0:
    continue
  xs = np.append(xs,x)
  ys = np.append(ys,y)

  xnbins = max(50,50*int(xsh[0]/250))
  ynbins = max(50,50*int(ysh[0]/250))

  k =kde.gaussian_kde([x,y])
  xi, yi = np.mgrid[x.min():x.max():xnbins*1j, y.min():y.max():ynbins*1j]
  zi = k(np.vstack([xi.flatten(), yi.flatten()]))
  plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.jet)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.colorbar()
  # plt.show()
  tag= str(well)+'_vs_'+x_label+'.png'
  plt.savefig(tag)
  plt.clf()

print('Processing final data ...')

xsh = np.shape(xs)
ysh = np.shape(ys)
xnbins = max(50, 50 * int(xsh[0] / 250))
ynbins = max(50, 50 * int(ysh[0] / 250))

k =kde.gaussian_kde([xs,ys])
xi, yi = np.mgrid[xs.min():xs.max():xnbins*1j, ys.min():ys.max():ynbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.jet)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.colorbar()
# plt.show()
tag= 'AllWells'+'_vs_'+x_label+'.png'
plt.savefig(tag)
plt.clf()


# df =  dfp[dfp['WELL_BORE_CODE']!=wellCode_p[-1]]
# wellCodeLabel = df['WELL_BORE_CODE']


#
# df = df.drop(['WELL_BORE_CODE'],axis=1)
# df = df.drop(['FLOW_KIND'],axis=1)
# df = df.fillna(0)
# df = df.replace({',': ''}, regex=True)
#
#
# df_y = df['BORE_OIL_VOL']
# y = df_y.values
# df_X = df.drop(['BORE_OIL_VOL'], axis=1)
# X = df_X.values1

# dataset = np.concatenate([np.asarray(X),np.asarray([y]).T], axis=1)
# dataset = dataset.astype('float32')
#
#
# # split into train and test sets
# train_size = int(len(dataset) * 0.8)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#
#
#
# print(len(train), len(test))



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

# # create and fit Multilayer Perceptron model
# from keras.models import Sequential
# from keras.layers import Dense
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
