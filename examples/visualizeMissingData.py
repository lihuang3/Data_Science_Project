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
rawdata = pandas.read_csv(os.path.join(dir,'data','VolveData.csv'), header=0,
            usecols=['DATEPRD', 'NPD_WELL_BORE_CODE','ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
                     'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P',
                      'DP_CHOKE_SIZE', 'BORE_OIL_VOL', 'FLOW_KIND', 'BORE_GAS_VOL', 'BORE_WAT_VOL','BORE_WI_VOL'], engine='python')



# dfi = dataframe[dataframe['FLOW_KIND'] == 'injection']
wellCode = np.unique(rawdata['NPD_WELL_BORE_CODE'])


vars_list = ['BORE_OIL_VOL','ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
             'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHT_P',
             'AVG_WHP_P', 'DP_CHOKE_SIZE', 'BORE_GAS_VOL', 'BORE_WAT_VOL','BORE_WI_VOL','DATEPRD']

num_features = 8
vars_mean_dict = {}

rawdata = rawdata.replace({',': ''}, regex=True)
label ='DATEPRD'
rawdates = rawdata[label].values
for id, it in enumerate(rawdates):
  res = it.split('-')
  rawdata[label].values[id] = int(res[2] + monthDict[res[1]] + res[0])
dates = rawdata[label].values
uniq_t = np.sort(np.unique(dates))
t_dict = {i:j for i,j in zip(uniq_t, np.arange(np.shape(uniq_t)[0]) )}

dataframe = rawdata[rawdata['ON_STREAM_HRS']>=0]

# we only consider production for now
dataframe = dataframe[dataframe['FLOW_KIND'] == 'production']


# Drop nan under 5%, else fill with -100 just for visualization
for x_label in vars_list[2:10]:
  tmp = dataframe[x_label]
  dropna = dataframe[tmp>=0]
  if np.shape(dropna)[0]/np.shape(tmp)[0] >=0.95:
    dataframe = dropna
  else:
    if (x_label == 'AVG_ANNULUS_PRESS'):
      dataframe[x_label].fillna(-10, inplace=True)
    else:
      dataframe[x_label].fillna(-50, inplace=True)

# Process date
x_label ='DATEPRD'
t = np.array(dataframe[x_label].values, dtype=np.int32)
for it in range(t.shape[0]):
  t[it] = t_dict[t[it]]

def outliers_z_score(vars, label):
  threshold = 3
  vars = vars.astype(np.float32)
  if (label=='AVG_DOWNHOLE_PRESSURE' or label == 'AVG_DOWNHOLE_TEMPERATURE'):
    vars_new = vars[vars>0]
    vzeros = vars<=1e-9
    mean = np.mean(vars_new)
    std = np.std(vars_new)
    z_scores = [(var - mean) / std for var in vars]
    outliers =  np.logical_and(vars>0,  np.abs(z_scores) > threshold)
    rest = np.logical_and( vars>0, np.abs(z_scores) <= threshold)

    mean = np.mean(vars[rest])
    std = np.std(vars[rest])
    z_scores = [(var - mean) / std for var in vars]
    outliers = np.logical_or( outliers, np.logical_and(rest, np.abs(z_scores) > threshold) )
    rest = np.logical_and(rest, (np.abs(z_scores) <= threshold) )
    vars[vzeros] = mean
    rest = np.logical_or(rest, vzeros, vars<0)

    return rest, vars
  else:
    mean = np.mean(vars[vars>=0])
    std = np.std(vars[vars>=0])
    z_scores = [(var-mean) / std for var in vars]

    outliers = np.abs(z_scores)>threshold
    rest = np.logical_or( np.abs(z_scores)<=threshold, vars<0)
    return rest, vars

from scipy.stats import gaussian_kde

# # Missing data visualization (vs Time)
# ================================================================
# f, axs =plt.subplots(4, 2)
#
# for it, x_label in enumerate(vars_list[2:10]):
#   x = np.array(dataframe[x_label].values, np.float32)
#   ts = np.copy(t)
#   # Outliers and missing data inputation
#   rest, vars = outliers_z_score(x, x_label)
#   x, ts = vars[rest], ts[rest]
#
#   tx = np.vstack([ts,x])
#   z = gaussian_kde(tx)(tx)
#   idx = z.argsort()
#   ts, x, z = ts[idx], x[idx], z[idx]
#   axs[int(it/2),int(it%2)].scatter(ts,x,c=z, s=10, edgecolor='')
#   axs[int(it / 2), int(it % 2)].set_title(x_label, fontsize = 16)
#   axs[int(it / 2), int(it % 2)].set_xlabel('time', fontsize = 16)
#   axs[int(it / 2), int(it % 2)].tick_params(labelsize=16)
# f.tight_layout()
# plt.show()
# -----------------------------------------------------------------


# # Features vs bore oil production data visualization
# ================================================================
# f, axs =plt.subplots(4, 2)
#
# x_label = vars_list[0]
# x = np.array(dataframe[x_label].values, np.float32)
# for it, y_label in enumerate(vars_list[2:10]):
#   ys = np.array(dataframe[y_label].values, np.float32)
#   xs = np.copy(x)
#   rest, vars = outliers_z_score(ys, y_label)
#   ys, xs = vars[rest], xs[rest]
#   xy = np.vstack([ys,xs])
#   z = gaussian_kde(xy)(xy)
#   idx = z.argsort()
#   xs, ys, z = xs[idx], ys[idx], z[idx]
#   axs[int(it/2),int(it%2)].scatter(ys,xs,c=z, s=10, edgecolor='')
#   axs[int(it / 2), int(it % 2)].set_title(x_label, fontsize = 16)
#   axs[int(it / 2), int(it % 2)].set_xlabel(y_label, fontsize = 16)
#   axs[int(it / 2), int(it % 2)].tick_params(labelsize=16)
# f.tight_layout()
# plt.show()
# -----------------------------------------------------------------

def scatter_plt(x, y, x_label, y_label, axs, figId, xlim=[], ylim=[]):
  xy = np.vstack([x, y])
  z = gaussian_kde(xy)(xy)
  idx = z.argsort()
  x, y, z = x[idx], y[idx], z[idx]

  axs[int(figId / 2), int(figId % 2)].scatter(x, y, c=z, s=10, edgecolor='')
  axs[int(figId / 2), int(figId % 2)].set_title(y_label, fontsize=16)
  axs[int(figId / 2), int(figId % 2)].set_xlabel(x_label, fontsize=16)
  axs[int(figId / 2), int(figId % 2)].tick_params(labelsize=16)
  if xlim:
    axs[int(figId / 2), int(figId % 2)].set_xlim([xlim[0], xlim[1]])
  if ylim:
    axs[int(figId / 2), int(figId % 2)].set_ylim([ylim[0], ylim[1]])

'''
Data Visualization second part: ON_STREAM_HRS & BORE_WI_VOL  
'''

df = rawdata.copy()
df = df[df['FLOW_KIND']=='injection']
df['BORE_WI_VOL'].fillna(-2000, inplace=True)


# Process date
x_label ='DATEPRD'
ti = np.array(df[x_label].values, dtype=np.int32)
for it in range(ti.shape[0]):
  ti[it] = t_dict[ti[it]]

y_label = vars_list[12]
f, axs =plt.subplots(1,2)

y = np.array(df[y_label].values, dtype=np.float32)
scatter_plt(ti, y, 'Time', y_label, axs, 0, [])
f.tight_layout()
plt.show()

# # Missing data visualization
# ================================================================
# f, axs =plt.subplots(3,2)
#
# y_label = 'BORE_OIL_VOL'
# it = 0
# y = np.array(dataframe[y_label].values, np.float32)
# scatter_plt(t, y, 'Time', y_label, axs, it)
#
# it = 4
# x_label = 'ON_STREAM_HRS'
# y_label = 'BORE_WI_VOL'
# x = np.array(df[x_label].values, np.float32)
# y = np.array(df[y_label].values, np.float32)
# scatter_plt(x,y, x_label, y_label, axs, it)
#
#
# x_label = 'ON_STREAM_HRS'
# it = 2
# x = np.array(df[x_label].values, np.float32)
# scatter_plt(ti, x, 'Time' ,x_label, axs, it)
#
#
# x_label = 'BORE_WI_VOL'
# it = 1
# x = np.array(df[x_label].values, np.float32)
# scatter_plt(ti, x, 'Time', x_label, axs, it)
#
# y_label = 'BORE_WI_VOL'
# for id, well in enumerate(well4inj):
#   y = np.array(df[df['NPD_WELL_BORE_CODE']==well][y_label].values, np.float32)
#   t1 = np.array(df[df['NPD_WELL_BORE_CODE']==well]['DATEPRD'].values, np.int32)
#   for it in range(t1.shape[0]):
#     t1[it] = t_dict[t1[it]]
#
#   it = 3+id*2
#   scatter_plt(t1, y, 'Time', y_label+'({})'.format(well), axs, it)
#
#
# f.tight_layout()
# plt.show()
# -----------------------------------------------------------------


# # Others vs bore oil production data visualization
# ================================================================
# f, axs =plt.subplots(2, 2)
# x_label = vars_list[0]
# x = np.array(dataframe[x_label].values, np.float32)
# from scipy.stats import gaussian_kde
# for it, y_label in enumerate(vars_list[1:2]):
#   y = np.array(dataframe[y_label].values, np.float32)
#   out, _ = outliers_z_score(y)
#   ys = y[out]
#   xs = x[out]
#   if it ==1:
#     xs, ys = x, y
#   xy = np.vstack([ys,xs])
#   z = gaussian_kde(xy)(xy)
#   idx = z.argsort()
#   xs, ys, z = xs[idx], ys[idx], z[idx]
#   axs[int(it/2),int(it%2)].scatter(ys,xs,c=z, s=10, edgecolor='')
#   axs[int(it / 2), int(it % 2)].set_ylabel(x_label)
#   axs[int(it / 2), int(it % 2)].set_xlabel(y_label)
# f.tight_layout()
# plt.show()
# -----------------------------------------------------------------



# f, axs =plt.subplots(3,2)
#
# y_label = vars_list[0]
# x_label ='DATEPRD'
#
# y = np.array(dataframe[y_label].values, dtype=np.float32)
# scatter_plt(t, y, 'Time', y_label, axs, 0, xlim=[0, 3400])
#
# id = 1
#
# for well in wellCode:
#   print('Processing well code {}'.format(well))
#   cur_well = dataframe[dataframe['NPD_WELL_BORE_CODE']==well]
#   ti = np.array(cur_well[x_label].values, dtype=np.int32)
#   for it in range(ti.shape[0]):
#     ti[it] = t_dict[ti[it]]
#   yi = np.array(cur_well[y_label].values, dtype=np.float32)
#   if yi.shape[0] == 0:
#     continue
#   scatter_plt(ti, yi, 'Time', str(well), axs, id, xlim=[0, 3400])
#   id += 1
#
# f.tight_layout()
# plt.show()


f, axs =plt.subplots(3,2)

x_label = vars_list[12]
y_label ='BORE_OIL_VOL'
x = np.array(dataframe[x_label].values, dtype=np.float32)
y = np.array(dataframe[y_label].values, dtype=np.float32)
scatter_plt(x, y, 'Time', x_label, axs, 0, [])

id = 1

for well in wellCode:
  print('Processing well code {}'.format(well))
  cur_well = dataframe[dataframe['NPD_WELL_BORE_CODE']==well]
  xi = np.array(cur_well[x_label].values, dtype=np.float32)
  yi = np.array(cur_well[y_label].values, dtype=np.float32)
  if yi.shape[0] == 0:
    continue
  scatter_plt(xi, yi, str(well), y_label, axs, id, [] )
  id += 1

f.tight_layout()
plt.show()


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
