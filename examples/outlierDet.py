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

  outliers = np.where(np.abs(z_scores)>threshold)
  rest = np.where(np.abs(z_scores)<=threshold)

  return rest, outliers


from scipy.stats import kde

xs, ys = np.zeros([]), np.zeros([])
y_label = vars_list[0]
x_label = vars_list[9]

print('Processing correlation b/w {} and {}'.format(y_label, x_label))

box_x1 = []
box_y1 = []
box_x2 = []
box_y2 = []

for id, well in enumerate(wellCode):
  # if well != 7289:
  #   continue
  print('Processing well code {}'.format(well))

  cur_well = dataframe[dataframe['NPD_WELL_BORE_CODE']==well]
  # if cur_well['FLOW_KIND'].values[0]=='injection':
  #   continue
  cur_well = cur_well[cur_well['FLOW_KIND']=='production']

  # if x_label =='DATEPRD':
  #   y = np.array(cur_well[y_label].values, dtype=np.float16)
  #   x = np.zeros(np.shape(y), dtype = np.int32)
  #
  #   tmp = cur_well[x_label].values
  #   for id, it in enumerate(tmp):
  #     res = it.split('-')
  #     x[id] = int(res[2] + monthDict[res[1]] + res[0])


  cur_well = cur_well[cur_well[x_label]>0]


  if cur_well.empty:
    continue
  bore_oil = np.array(cur_well[y_label].values, dtype=np.float16)
  x = np.array(cur_well[x_label].values, dtype=np.float16)

  out, _ = outliers_z_score(x)
  x2 = x[out]
  x1 = x

  box_y1.extend(x1)
  box_y2.extend(x2)

  box_x1.extend(well*np.ones(shape = np.shape(x1), dtype=np.uint8))
  box_x2.extend(well*np.ones(shape = np.shape(x2), dtype=np.uint8))
  # NaN data cleaning for var_list[9] 'DP_CHOKE_SIZE'
  # where NaN data are replaced by neighbor mean values
  nanData = np.where(np.isnan(x))
  if nanData[0].size>0:
    print('Error! Terminating program ...')
    exit()

  xs = np.append(xs,x)
  ys = np.append(ys,bore_oil)


  # # plot and save
  # sns.boxplot(y=x)
  # plt.show()
  # print('')
  # tag= str(well)+'_box_'+x_label+'.png'
  # plt.savefig(tag)
  # plt.clf()

print('Processing final data ...')

f, (ax1, ax2) =plt.subplots(1, 2, sharey=True)


sns.boxenplot(y=box_y1,x=box_x1, ax=ax1)

ax1.set_xlabel('Well Code')
ax1.set_ylabel(x_label)
ax1.set_title('w/o z-scores')

sns.boxenplot(y=box_y2,x=box_x2, ax=ax2)
# plt.ylabel(x_label)
ax2.set_xlabel('Well Code')
ax2.set_title('w/ z-scores')


# plt.show()
# # plot and save
# sns.boxplot(xs)
# plt.show()
tag= 'AllWells'+'zscore_'+x_label+'.png'
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
