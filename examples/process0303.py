import sys, os, math
dir =  os.getcwd()

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.cluster import KMeans
import pickle
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
             'AVG_WHP_P', 'DP_CHOKE_SIZE', 'BORE_GAS_VOL', 'BORE_WAT_VOL','BORE_WI_VOL', 'BORE_WI_VOL','DATEPRD']

well_dict = {7405: '1C', 7078: '11H', 5599: '12H', 5351: '14H', 7289: '15D', 5693: '4AH', 5769: '5AH'}

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


df = rawdata[ rawdata['FLOW_KIND'] == 'production']
# Divide production data into produtive and non-productive subsets
# productive df1
df1 = df[np.array(df['BORE_OIL_VOL'].values, np.int32) > 10]
# non-productive df2
df2 = df[ ~ (np.array(df['BORE_OIL_VOL'].values, np.int32) > 10) ]


# Drop na under 5% bu subset, else fill with negative constant just for visualization
for x_label in vars_list[1:10]:
  tmp = df1[x_label]
  nums = df1[tmp>=0]
  if np.shape(nums)[0]/np.shape(tmp)[0] >=0.95:
    df1 = nums
  else:
    if (x_label == 'AVG_ANNULUS_PRESS'):
      df1[x_label].fillna(-10, inplace=True)
    else:
      df1[x_label].fillna(-50, inplace=True)

# Process date
x_label ='DATEPRD'
t = np.array(df1[x_label].values, dtype=np.int32)
for it in range(t.shape[0]):
  t[it] = t_dict[t[it]]

'''
Use z-score to remove outliers
'''
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

# # Data visualization (all features vs Time, w/o filtering outliers)
# ================================================================
# f, axs =plt.subplots(4, 2)
#
# for it, x_label in enumerate(vars_list[2:10]):
#   x = np.array(df1[x_label].values, np.float32)
#   ts = np.copy(t)
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

# # Data visualization (vs Time, w/o filtering outliers)

"""
Rules: 
1. drop NAs if num of NAs is less than 5% of the total data
2. Imput 0s with overall mean for `AVG_DOWNHOLE_PRESS` and `AVG_DOWNHOLE_PRESS`
3. Others, perform z-score for all other features
"""
# ================================================================
# f, axs =plt.subplots(4, 2)
#
# for it, x_label in enumerate(vars_list[2:10]):
#
#
#   x = np.array(df1[x_label].values, np.float32)
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


"""
Each well feature vs time
"""
# for y_label in vars_list[2:10]:
  # f, axs = plt.subplots(3, 2, figsize=(16, 12))
  # print('\nProcessing %s' % (y_label))
  # x_label = 'DATEPRD'
  # y = np.array(df1[y_label].values, dtype=np.float32)
  # ylim = [int(y.min()) - 10, int(y.max()) + 10]
  # scatter_plt(t, y, 'Time', y_label, axs, 0, xlim=[0, 3400], ylim=ylim)
  #
  # id = 1
  # for well in wellCode:
  #   print('Processing well code {}'.format(well))
  #   cur_well = df1[df1['NPD_WELL_BORE_CODE'] == well]
  #   ti = np.array(cur_well[x_label].values, dtype=np.int32)
  #   for it in range(ti.shape[0]):
  #     ti[it] = t_dict[ti[it]]
  #   yi = np.array(cur_well[y_label].values, dtype=np.float32)
  #   if yi.shape[0] == 0:
  #     continue
  #
  #   scatter_plt(ti, yi, 'Time', str(well_dict[well]), axs, id, xlim=[0, 3400], ylim=ylim)
  #
  #   id += 1
  #
  # f.tight_layout()
  # f.subplots_adjust(top=0.956,
  #                   bottom=0.082,
  #                   left=0.042,
  #                   right=0.988,
  #                   hspace=0.506,
  #                   wspace=0.098)
  # tag = y_label + '_vs_' + 'time.png'
  # figpath = os.path.join('df1fig1', tag)
  # plt.savefig(figpath)
  # plt.clf()

"""
Rules: 
1. drop NAs if num of NAs is less than 5% of the total data
2. Imput 0s with well mean for `AVG_DOWNHOLE_PRESS` and `AVG_DOWNHOLE_PRESS`
3. Others, perform z-score on each well
"""
# for y_label in vars_list[2:10]:
#   f, axs = plt.subplots(3, 2, figsize=(16, 12))
#   print('\nProcessing %s'%(y_label))
#   x_label ='DATEPRD'
#   y = np.array(df1[y_label].values, dtype=np.float32)
#   ylim = [int(y.min())-10, int(y.max())+10]
#
#   ytmp, ttmp = np.empty((0)), np.empty((0))
#
#   # scatter_plt(t, y, 'Time', y_label, axs, 0, xlim=[0, 3400], ylim=ylim)
#
#   id = 1
#   for well in wellCode:
#     print('Processing well code {}'.format(well))
#     cur_well = df1[df1['NPD_WELL_BORE_CODE']==well]
#     ti = np.array(cur_well[x_label].values, dtype=np.int32)
#     for it in range(ti.shape[0]):
#       ti[it] = t_dict[ti[it]]
#     yi = np.array(cur_well[y_label].values, dtype=np.float32)
#     if yi.shape[0] == 0:
#       continue
#
#     # # Outliers and missing data inputation
#     rest, vars = outliers_z_score(yi, y_label)
#     yi, ti = vars[rest], ti[rest]
#     ytmp, ttmp = np.append(ytmp, yi), np.append(ttmp, ti)
#
#
#     scatter_plt(ti, yi, 'Time', str(well_dict[well]), axs, id, xlim=[0, 3400], ylim=ylim)
#
#     id += 1
#
#   scatter_plt(ttmp, ytmp, 'Time', y_label, axs, 0, xlim=[0, 3400], ylim=ylim)
#   f.tight_layout()
#   f.subplots_adjust(top=0.956,
#   bottom=0.082,
#   left=0.042,
#   right=0.988,
#   hspace=0.506,
#   wspace=0.098)
#   tag= y_label + '_vs_' + 'time_z.png'
#   figpath = os.path.join('df1fig1', tag)
#   plt.savefig(figpath)
#   plt.clf()

"""
Rules:
1. Imput 0s in `AVG_DOWNHOLE_PRESS` and `AVG_DOWNHOLE_PRESS`, 
2. first remove all outliers by wells for each features other than these two.
3. secondly, remove the non-zero outliers in these two features
4. thirdly,  make zero-value rows of these two features as test set, others as the training set
"""
def outliers_z_score2(vars, label):
  threshold = 3
  vars = vars.astype(np.float32)
  if label in ['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE']:
    normal = vars>0
    outliers2 = ~ normal
  elif label == 'AVG_ANNULUS_PRESS':
    normal = vars>=5
    outliers2 = ~normal
    return vars, normal, vars<-100, outliers2
  else:
    normal = vars>=0
    outliers2 = ~ normal

  if label == 'ON_STREAM_HRS':
    vars[vars>24] = 24
    to_imput = np.where(vars <= 0.1)
    for idx in to_imput[0]:
      neighbors = []
      rng = 1
      while len(neighbors)<2:
        if idx+rng < vars.shape[0] and vars[idx+rng] > 0.1:
          neighbors.append(vars[idx+rng])
        if idx-rng >= 0 and vars[idx-rng] > 0.1:
          neighbors.append(vars[idx-rng])
        rng += 1
      vars[idx] = np.mean(neighbors)
    return vars, normal, ~normal, outliers2

  if label in ['AVG_WHP_P']:
    to_imput = np.where(vars <= 10)
    for idx in to_imput[0]:
      neighbors = []
      rng = 1
      while len(neighbors)<2:
        if idx+rng < vars.shape[0] and vars[idx+rng] > 10:
          neighbors.append(vars[idx+rng])
        if idx-rng >= 0 and vars[idx-rng] > 10:
          neighbors.append(vars[idx-rng])
        rng += 1
      vars[idx] = np.mean(neighbors)


  vars_new = vars[normal]
  mean = np.mean(vars_new)
  std = np.std(vars_new)
  z_scores = [(var - mean) / std for var in vars]
  outliers =  np.logical_and(normal,  np.abs(z_scores) > threshold)
  filtered = np.logical_and( normal, np.abs(z_scores) <= threshold)

  return vars, filtered, outliers, outliers2

test_idx_dhp = np.empty((0), dtype=np.int32)
test_idx_dht = np.empty((0), dtype=np.int32)
test_idx_dptb = np.empty((0), dtype=np.int32)
test_idx_ann = np.empty((0), dtype=np.int32)

idx_to_del = np.empty((0),dtype=np.int32)



# y = np.array(df1['AVG_ANNULUS_PRESS'].values, dtype=np.float32)
# hist = np.histogram(y, bins=10, range=(1e-3, y.max()) )

for y_label in vars_list[1:10]:
  # if y_label in ['AVG_ANNULUS_PRESS']:
  #   continue
  # f, axs = plt.subplots(3, 2, figsize=(16, 12))

  # print('\nProcessing %s'%(y_label))
  x_label ='DATEPRD'
  y = np.array(df1[y_label].values, dtype=np.float32)
  if y_label == 'AVG_DP_TUBING':
    kmeans = KMeans(n_clusters=2, random_state=0).fit(y.reshape(-1, 1))
    if np.where(kmeans.labels_ == 0)[0].size < 0.5 * kmeans.labels_.size:
      normal = kmeans.labels_ == 1
    else:
      normal = kmeans.labels_ == 0
    y[~normal] = -10.0
    df1[y_label].values[~normal] = -10.0

  ylim = [int(y.min())-10, int(y.max())+10]
  ytmp, ttmp = np.empty((0)), np.empty((0))

  id = 1
  for well in wellCode:
    # print('Processing well code {}'.format(well))
    idx = np.where(df1['NPD_WELL_BORE_CODE']==well)
    cur_well = df1[df1['NPD_WELL_BORE_CODE']==well]
    ti = np.array(cur_well[x_label].values, dtype=np.int32)
    for it in range(ti.shape[0]):
      ti[it] = t_dict[ti[it]]
    yi = np.array(cur_well[y_label].values, dtype=np.float32)
    if yi.shape[0] == 0:
      continue

    # # Outliers and missing data imputation
    yi, filtered, outliers, outliers2 = outliers_z_score2(yi, y_label)
    if y_label == 'AVG_ANNULUS_PRESS':
      test_idx_ann = np.append(test_idx_ann, idx[0][outliers2])
    if y_label == 'AVG_DOWNHOLE_PRESSURE':
      test_idx_dhp = np.append(test_idx_dhp, idx[0][outliers2])
    if y_label == 'AVG_DOWNHOLE_TEMPERATURE':
      test_idx_dht = np.append(test_idx_dht, idx[0][outliers2])
    if y_label == 'AVG_DP_TUBING':
      test_idx_dptb = np.append(test_idx_dptb, idx[0][outliers2])

    # train_idx = np.append(train_idx, idx[0][filtered])
    idx_to_del = np.append(idx_to_del, idx[0][outliers])
    yi, ti = yi[np.logical_or(filtered, outliers2)], ti[np.logical_or(filtered, outliers2)]
    ytmp, ttmp = np.append(ytmp, yi), np.append(ttmp, ti)

  #   scatter_plt(ti, yi, 'Time', str(well_dict[well]), axs, id, xlim=[0, 3400], ylim=ylim)
  #   id += 1
  #
  # scatter_plt(ttmp, ytmp, 'Time', y_label, axs, 0, xlim=[0, 3400], ylim=ylim)
  # f.tight_layout()
  # f.subplots_adjust(top=0.956,
  # bottom=0.082,
  # left=0.042,
  # right=0.988,
  # hspace=0.506,
  # wspace=0.098)
  # tag= y_label + '_vs_' + 'time_z.png'
  # figpath = os.path.join('df1fig2', tag)
  # plt.savefig(figpath)
  # plt.clf()


## Test set indices
test_idx_dhp = np.unique(test_idx_dhp)
test_idx_dht = np.unique(test_idx_dht)
test_idx_ann = np.unique(test_idx_ann)
test_idx_dptb = np.unique(test_idx_dptb)

# train_idx = np.unique(train_idx)

## Outliers to be removed from datatframe
idx_to_del = np.unique(idx_to_del)

# irrelevant features
irrelevant_features = ['DATEPRD', 'NPD_WELL_BORE_CODE',
                       'FLOW_KIND', 'BORE_GAS_VOL',
                       'BORE_WAT_VOL','BORE_WI_VOL']

relevant_features = ['AVG_ANNULUS_PRESS', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DOWNHOLE_PRESSURE',
                     'ON_STREAM_HRS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P',
                     'AVG_WHT_P', 'DP_CHOKE_SIZE', 'AVG_DP_TUBING',
                     'BORE_OIL_VOL']

df11 = df1.drop(columns=irrelevant_features, inplace=False)
num_features = len(relevant_features)
data_size = len(df11[relevant_features[0]].values)
dataset = np.empty((data_size, num_features))
for idx, feature in enumerate(relevant_features):
  dataset[:, idx] = np.array(df11[feature].values, dtype=np.float32)


def SVR_imput(dataset, idx_to_del, selected_features, imput_feature,
              target, test_idx, gamma, cost, test_ratio=0.1, run_train=False):
  # split into train and test sets

  num_features = len(selected_features)

  train_idx = np.ones(dataset.shape[0], dtype=np.uint8)
  train_idx[np.unique(np.append(test_idx, idx_to_del))] = 0

  train_set = dataset[train_idx > 0, :]
  test_set = dataset[test_idx, :]

  # Normalization
  mean, std = np.mean(train_set, axis=0), np.std(train_set, axis=0)
  train_set = (train_set - mean) / (std + 1e-5)
  test_set = (test_set - mean) / (std + 1e-5)

  print(imput_feature, ': \n', 'training set ', train_set.shape[0], 'test set ', test_set.shape[0])

  train_X, test_X, train_y, test_y = \
    train_set[:, selected_features], test_set[:, selected_features], \
    train_set[:, target], test_set[:, target]
  train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=test_ratio)

  # Method 1 SVR
  if not run_train and os.path.isfile( os.path.join(dir, 'models', '%s_svr.weights'%(imput_feature))):
    print('Loading saved %s weights...'%(imput_feature))
    clf = pickle.load(open(os.path.join(dir, 'models', '%s_svr.weights'%(imput_feature)),'rb'))
  else:
    clf = svm.SVR(kernel='rbf', gamma=gamma, C=cost)
    clf.fit(train_X, train_y)
    pickle.dump(clf, open(os.path.join(dir, 'models', '%s_svr.weights'%(imput_feature)), 'wb'))

  confidence = clf.score(val_X, val_y)
  print('rbf', imput_feature, confidence, '\n')

  yhat = clf.predict(test_X)
  dataset[test_idx, target] = np.copy(yhat * std[target] + mean[target])


SVR_imput(dataset, idx_to_del, selected_features=[3, 4, 5, 6, 7],
          imput_feature='AVG_DP_TUBING', target=8, test_idx=test_idx_dptb,
          gamma=3, cost=5, test_ratio=0.2)

SVR_imput(dataset, idx_to_del, selected_features=[3, 4, 5, 6, 7],
          imput_feature='AVG_DOWNHOLE_TEMPERATURE', target=2, test_idx=test_idx_dht,
          gamma=4, cost=2, test_ratio=0.2, run_train=True) # 0.01 10

SVR_imput(dataset, idx_to_del, selected_features=[3, 4, 5, 6, 7],
          imput_feature='AVG_DOWNHOLE_PRESSURE', target=1, test_idx=test_idx_dhp,
          gamma=1.4, cost=20, test_ratio=0.2)

SVR_imput(dataset, idx_to_del, selected_features=[1,2, 3, 4, 5, 6, 7, 8],
          imput_feature='ANNULUS_PRESS', target=0, test_idx=test_idx_ann,
          gamma=7, cost=30, test_ratio=0.2)


selected = [0, 1, 2, 3, 4, 5, 6, 7, 8]
num_features = len(selected)

imput_feature, target = 'BORE_OIL_VOL', 9

train_X = dataset[:, selected]
train_y = dataset[:, target]

train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2)

# Normalization
Xmean , Xstd = np.mean(train_X, axis=0), np.std(train_X, axis=0)
train_X = (train_X - Xmean ) / (Xstd + 1e-5)
test_X = (test_X - Xmean) / (Xstd + 1e-5)

ymean , ystd = np.mean(train_y, axis=0), np.std(train_y, axis=0)
train_y = (train_y - ymean ) / (ystd + 1e-5)
test_y = (test_y - ymean) / (ystd + 1e-5)


print('\n',train_X.shape[0], test_X.shape[0])

# Method 1 SVR
if os.path.isfile(os.path.join(dir, 'models', '%s_svr.weights' % (imput_feature))):
  print('Loading saved %s weights...' % (imput_feature))
  clf = pickle.load(open(os.path.join(dir, 'models', '%s_svr.weights' % (imput_feature)), 'rb'))
else:
  clf = svm.SVR(kernel='rbf', gamma=0.16, C=50)
  clf.fit(train_X, train_y)
  pickle.dump(clf, open(os.path.join(dir, 'models', '%s_svr.weights' % (imput_feature)), 'wb'))

yhat = clf.predict(test_X)
confidence = clf.score(test_X, test_y)
print('rbf', imput_feature, confidence)



def scatter_plt2(x1, y1, x2, y2, x_label, y_label, axs, figId, xlim=[], ylim=[]):

  axs[int(figId / 2), int(figId % 2)].scatter(x1, y1, c='b', s=10, edgecolor='')
  axs[int(figId / 2), int(figId % 2)].scatter(x2, y2, c='r', s=10, edgecolor='')
  axs[int(figId / 2), int(figId % 2)].set_title(y_label, fontsize=16)
  axs[int(figId / 2), int(figId % 2)].set_xlabel(x_label, fontsize=16)
  axs[int(figId / 2), int(figId % 2)].tick_params(labelsize=16)
  if xlim:
    axs[int(figId / 2), int(figId % 2)].set_xlim([xlim[0], xlim[1]])
  if ylim:
    axs[int(figId / 2), int(figId % 2)].set_ylim([ylim[0], ylim[1]])


['DATEPRD', 'NPD_WELL_BORE_CODE']

x_label = 'DATEPRD'

for y_label in vars_list[1:10]:

  f, axs = plt.subplots(3, 2, figsize=(16, 12))
  y = np.array(df1[y_label].values, dtype=np.float32)
  ylim = [int(y.min())-10, int(y.max())+10]
  ytmp, ttmp = np.empty((0)), np.empty((0))
  id = 1

  for well in wellCode:
    idx = np.where(df1['NPD_WELL_BORE_CODE']==well)
    cur_well = df1[df1['NPD_WELL_BORE_CODE']==well]
    ti = np.array(cur_well[x_label].values, dtype=np.int32)
    for it in range(ti.shape[0]):
      ti[it] = t_dict[ti[it]]
    yi = np.array(cur_well[y_label].values, dtype=np.float32)
    if yi.shape[0] == 0:
      continue

    # # Outliers and missing data inputation
    yi, filtered, outliers, outliers2 = outliers_z_score2(yi, y_label)
    if y_label == 'AVG_ANNULUS_PRESS':
      df1
      test_idx_ann = np.append(test_idx_ann, idx[0][outliers2])
    if y_label == 'AVG_DOWNHOLE_PRESSURE':
      test_idx_dhp = np.append(test_idx_dhp, idx[0][outliers2])
    if y_label == 'AVG_DOWNHOLE_TEMPERATURE':
      test_idx_dht = np.append(test_idx_dht, idx[0][outliers2])
    if y_label == 'AVG_DP_TUBING':
      test_idx_dptb = np.append(test_idx_dptb, idx[0][outliers2])

    # train_idx = np.append(train_idx, idx[0][filtered])
    idx_to_del = np.append(idx_to_del, idx[0][outliers])
    yi, ti = yi[np.logical_or(filtered, outliers2)], ti[np.logical_or(filtered, outliers2)]
    ytmp, ttmp = np.append(ytmp, yi), np.append(ttmp, ti)

  #   scatter_plt(ti, yi, 'Time', str(well_dict[well]), axs, id, xlim=[0, 3400], ylim=ylim)
  #   id += 1
  #
  # scatter_plt(ttmp, ytmp, 'Time', y_label, axs, 0, xlim=[0, 3400], ylim=ylim)
  # f.tight_layout()
  # f.subplots_adjust(top=0.956,
  # bottom=0.082,
  # left=0.042,
  # right=0.988,
  # hspace=0.506,
  # wspace=0.098)
  # tag= y_label + '_vs_' + 'time_z.png'
  # figpath = os.path.join('df1fig2', tag)
  # plt.savefig(figpath)
  # plt.clf()


# # create and fit Multilayer Perceptron model
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
#
# model = Sequential()
# model.add(Dense(16, activation='relu', input_dim=num_features))
# # model.add(Dropout(0.5))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))
#
# model.compile(optimizer='adam', loss='mse')
# # fit model
# res = model.fit(train_X, train_y, validation_split=0.1, epochs=100, verbose=2)
# # demonstrate prediction
#
# yhat = model.predict(test_X, verbose=0)
# # print(np.mean(np.abs(yhat-test_y)/np.abs(test_y)))
#
#
# # # Estimate model performance
# trainScore = model.evaluate(train_X, train_y, verbose=0)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
# # testScore = model.evaluate(test_X, test_y, verbose=0)
# # print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
#
#
#
# # summarize history for loss
#
#
# plt.plot(res.history['loss'])
# plt.plot(res.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()



for y_label in vars_list[1:10]:

  # f, axs = plt.subplots(3, 2, figsize=(16, 12))

  print('\nProcessing %s'%(y_label))
  x_label ='DATEPRD'
  y = np.array(df1[y_label].values, dtype=np.float32)
  ylim = [int(y.min())-10, int(y.max())+10]

  ytmp, ttmp = np.empty((0)), np.empty((0))

  id = 1
  for well in wellCode:
    print('Processing well code {}'.format(well))
    idx = np.where(df1['NPD_WELL_BORE_CODE']==well)
    cur_well = df1[df1['NPD_WELL_BORE_CODE']==well]
    ti = np.array(cur_well[x_label].values, dtype=np.int32)
    for it in range(ti.shape[0]):
      ti[it] = t_dict[ti[it]]
    yi = np.array(cur_well[y_label].values, dtype=np.float32)
    if yi.shape[0] == 0:
      continue

    # # Outliers and missing data inputation
    yi, filtered, outliers, outliers2 = outliers_z_score2(yi, y_label)
    if y_label in ['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE']:
      test_idx = np.append(test_idx, idx[0][outliers2])
    # train_idx = np.append(train_idx, idx[0][filtered])
    idx_to_del = np.append(idx_to_del, idx[0][outliers])
    yi, ti = yi[np.logical_or(filtered, outliers2)], ti[np.logical_or(filtered, outliers2)]
    ytmp, ttmp = np.append(ytmp, yi), np.append(ttmp, ti)


"""
Each well feature vs bore oil vol
"""
#
# for x_label in vars_list[2:10]:
#
#   f, axs = plt.subplots(3, 2, figsize=(16, 12))
#
#   print('\nProcessing %s'%(x_label))
#   y_label = 'BORE_OIL_VOL'
#
#   x = np.array(df1[x_label].values, dtype=np.float32)
#   y = np.array(df1[y_label].values, dtype=np.float32)
#   ylim = [int(y.min())-10, int(y.max())+10]
#   xlim = [int(x.min())-10, int(x.max())+10]
#   scatter_plt(x, y, x_label, y_label, axs, 0, xlim=xlim, ylim=ylim)
#
#   id = 1
#   for well in wellCode:
#     print('Processing well code {}'.format(well))
#     cur_well = df1[df1['NPD_WELL_BORE_CODE']==well]
#     xi = np.array(cur_well[x_label].values, dtype=np.float32)
#     yi = np.array(cur_well[y_label].values, dtype=np.float32)
#     if yi.shape[0] == 0:
#       continue
#     scatter_plt(xi, yi, x_label, str(well_dict[well]), axs, id, xlim=xlim, ylim=ylim)
#     id += 1
#
#   f.tight_layout()
#   f.subplots_adjust(top=0.956,
#   bottom=0.082,
#   left=0.042,
#   right=0.988,
#   hspace=0.506,
#   wspace=0.098)
#   tag= x_label + '_vs_' + 'oil.png'
#   figpath = os.path.join('df1fig1', tag)
#   plt.savefig(figpath)
#   plt.clf()

"""
Rules: 
1. drop NAs if num of NAs is less than 5% of the total data
2. Imput 0s with well mean for `AVG_DOWNHOLE_PRESS` and `AVG_DOWNHOLE_PRESS`
3. Others, perform z-score on each well
"""
# for x_label in vars_list[2:10]:
#
#   f, axs = plt.subplots(3, 2, figsize=(16, 12))
#
#   print('\nProcessing %s'%(x_label))
#   y_label = 'BORE_OIL_VOL'
#
#   x = np.array(df1[x_label].values, dtype=np.float32)
#   y = np.array(df1[y_label].values, dtype=np.float32)
#   ylim = [int(y.min())-10, int(y.max())+10]
#   xlim = [int(x.min())-10, int(x.max())+10]
#
#   xtmp, ytmp = np.empty((0)), np.empty((0))
#
#   id = 1
#   for well in wellCode:
#     print('Processing well code {}'.format(well))
#     cur_well = df1[df1['NPD_WELL_BORE_CODE']==well]
#     xi = np.array(cur_well[x_label].values, dtype=np.float32)
#     yi = np.array(cur_well[y_label].values, dtype=np.float32)
#     if yi.shape[0] == 0:
#       continue
#     # Outliers and missing data inputation
#     rest, vars = outliers_z_score(xi, x_label)
#     xi, yi = vars[rest], yi[rest]
#     xtmp, ytmp = np.append(xtmp, xi), np.append(ytmp, yi)
#
#     scatter_plt(xi, yi, x_label, str(well_dict[well]), axs, id, xlim=xlim, ylim=ylim)
#     id += 1
#
#   scatter_plt(xtmp, ytmp, x_label, str(well_dict[well]), axs, 0, xlim=xlim, ylim=ylim)
#
#   f.tight_layout()
#   f.subplots_adjust(top=0.956,
#   bottom=0.082,
#   left=0.042,
#   right=0.988,
#   hspace=0.506,
#   wspace=0.098)
#   tag= x_label + '_vs_' + 'oilz.png'
#   figpath = os.path.join('df1fig1', tag)
#   plt.savefig(figpath)
#   plt.clf()

"""
Rules:
1. Imput 0s in `AVG_DOWNHOLE_PRESS` and `AVG_DOWNHOLE_PRESS`, 
2. first remove all outliers by wells for each features other than these two.
3. secondly, remove the non-zero outliers in these two features
4. thirdly,  make zero-value rows of these two features as test set, others as the training set
"""

# test_idx = np.empty((0), dtype=np.int32)
# idx_to_del = np.empty((0),dtype=np.int32)
#
# for x_label in vars_list[1:10]:
#   if x_label in ['DP_CHOKE_SIZE', 'AVG_ANNULUS_PRESS']:
#     continue
#
#   f, axs = plt.subplots(3, 2, figsize=(16, 12))
#   print('\nProcessing %s'%(x_label))
#   y_label = 'BORE_OIL_VOL'
#
#   x = np.array(df1[x_label].values, dtype=np.float32)
#   y = np.array(df1[y_label].values, dtype=np.float32)
#   ylim = [int(y.min())-0.1*y.max(), int(y.max())+0.1*y.max()]
#   xlim = [int(x.min())-0.1*x.max(), int(x.max())+0.1*x.max()]
#
#   ytmp, xtmp = np.empty((0)), np.empty((0))
#
#   id = 1
#   for well in wellCode:
#     print('Processing well code {}'.format(well))
#     idx = np.where(df1['NPD_WELL_BORE_CODE']==well)
#     cur_well = df1[df1['NPD_WELL_BORE_CODE']==well]
#     xi = np.array(cur_well[x_label].values, dtype=np.float32)
#     yi = np.array(cur_well[y_label].values, dtype=np.float32)
#     if yi.shape[0] == 0:
#       continue
#
#     # # Outliers and missing data inputation
#     xi, filtered, outliers, outliers2 = outliers_z_score2(xi, x_label)
#     if y_label in ['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE']:
#       test_idx = np.append(test_idx, idx[0][outliers2])
#     idx_to_del = np.append(idx_to_del, idx[0][outliers])
#     yi, xi = yi[np.logical_or(filtered, outliers2)], xi[np.logical_or(filtered, outliers2)]
#     ytmp, xtmp = np.append(ytmp, yi), np.append(xtmp, xi)
#
#     scatter_plt(xi, yi, x_label, str(well_dict[well]), axs, id, xlim=xlim, ylim=ylim)
#     id += 1
#
#   scatter_plt(xtmp, ytmp, x_label, str(well_dict[well]), axs, 0, xlim=xlim, ylim=ylim)
#   f.tight_layout()
#   f.subplots_adjust(top=0.956,
#   bottom=0.082,
#   left=0.042,
#   right=0.988,
#   hspace=0.506,
#   wspace=0.098)
#   tag= x_label + '_vs_' + 'oil_z.png'
#   figpath = os.path.join('df1fig2', tag)
#   plt.savefig(figpath)
#   plt.clf()



'''
Data Visualization second part: ON_STREAM_HRS & BORE_WI_VOL  
'''

df = rawdata.copy()
df = df[df['FLOW_KIND']=='injection']
df['ON_STREAM_HRS'].fillna(-10, inplace=True)
df['BORE_WI_VOL'].fillna(-2000, inplace=True)
well4inj = np.unique(df['NPD_WELL_BORE_CODE'])

# Process date
x_label ='DATEPRD'
ti = np.array(df[x_label].values, dtype=np.int32)
for it in range(ti.shape[0]):
  ti[it] = t_dict[ti[it]]



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






f, axs =plt.subplots(3,2)

x_label = vars_list[9]
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
