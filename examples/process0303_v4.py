import sys, os, math
dir =  os.getcwd()

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
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

rawdata = pandas.ExcelFile(os.path.join(dir,'data','datasets.xlsx'))

vars_list = ['BORE_OIL_VOL','ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
             'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHT_P',
             'AVG_WHP_P', 'DP_CHOKE_SIZE', 'BORE_GAS_VOL', 'BORE_WAT_VOL','BORE_WI_VOL', 'BORE_WI_VOL','DATEPRD']

well_dict = {7405: '1C', 7078: '11H', 5599: '12H', 5351: '14H', 7289: '15D', 5693: '4AH', 5769: '5AH'}

num_features = 8
vars_mean_dict = {}

label = 'DATEPRD'
sheet_name = rawdata.sheet_names
sheet = rawdata.parse(sheet_name=sheet_name[0])
rawdates = pandas.to_datetime(sheet[label].values)
sheet[label].values.dtype = np.int64
for id, it in enumerate(rawdates):
  sheet[label].values[id] = 10000*rawdates[id].year + 100*rawdates[id].month + rawdates[id].day


dates = np.array(sheet[label].values, dtype=np.int64)
uniq_t = np.sort(np.unique(dates))
t_dict = {i:j for i,j in zip(uniq_t, np.arange(np.shape(uniq_t)[0]) )}


# irrelevant features
irrelevant_features = ['DATEPRD', 'WELL_BORE_CODE',
                       'FLOW_KIND', 'BORE_GAS_VOL',
                       'BORE_WAT_VOL']

relevant_features = ['class',  'AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE',
                     'AVG_DOWNHOLE_PRESSURE', 'AVG_ANNULUS_PRESS',
                     'ON_STREAM_HRS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P',
                     'AVG_WHT_P', 'DP_CHOKE_SIZE',
                     'BORE_OIL_VOL']

num_features = len(relevant_features)
data_size = len(sheet[relevant_features[0]].values)
dataset = np.empty((data_size, num_features))
for idx, feature in enumerate(relevant_features):
  dataset[:, idx] = np.array(sheet[feature].values, dtype=np.float32)


def SVR_imput(dataset, class_idx, selected_features, imput_feature,
              target, gamma, cost, run_train=False):
  # split into train and test sets

  num_features = len(selected_features)

  train_set = dataset[class_idx>2]
  test_set = dataset[np.logical_and(class_idx<=2, class_idx>=1)]
  missing_set = dataset[class_idx==0]

  # Normalization
  mean, std = np.mean(train_set, axis=0), np.std(train_set, axis=0)
  train_set = (train_set - mean) / (std + 1e-6)
  test_set = (test_set - mean) / (std + 1e-6)
  missing_set = (missing_set - mean) / (std + 1e-6)
  print(imput_feature, ': \n', 'training set ', train_set.shape[0], 'test set ', test_set.shape[0])

  train_X, test_X, train_y, test_y = \
    train_set[:, selected_features], test_set[:, selected_features], \
    train_set[:, target], test_set[:, target]

  train_X, train_y = shuffle(train_X, train_y)

  pred_X, pred_y = missing_set[:, selected_features], missing_set[:, target]


  if not run_train and os.path.isfile( os.path.join(dir, 'models', '%s_svr.weights'%(imput_feature))):
    print('Loading saved %s weights...'%(imput_feature))
    clf = pickle.load(open(os.path.join(dir, 'models', '%s_svr.weights'%(imput_feature)),'rb'))
  else:
    clf = svm.SVR(kernel='rbf', gamma=gamma, C=cost)
    cvs = cross_val_score(clf, train_X, train_y, cv=10)
    conf_mean, conf_std = np.mean(cvs), np.std(cvs)
    print('rbf', imput_feature, ' cross validation: ', '%.3f' % (conf_mean), '%.4f' % (conf_std), '\n')
    clf.fit(train_X, train_y)

    # all_X = np.append(train_X, test_X, axis=0)
    # all_y = np.append(train_y, test_y, axis=0)
    # all_X, all_y = shuffle(all_X, all_y)
    # clf.fit(all_X, all_y)
    pickle.dump(clf, open(os.path.join(dir, 'models', '%s_svr.weights'%(imput_feature)), 'wb'))

  confidence = clf.score(test_X, test_y)
  print('rbf', imput_feature, '%.3f' % (confidence), '\n')

  if not imput_feature == 'BORE_OIL_VOL':
    pred_y = clf.predict(pred_X) * std[target] + mean[target]
    dataset[class_idx==0, target] = np.copy(pred_y)


class_idx = np.array(sheet['class'].values, dtype=np.int16)
SVR_imput(dataset, class_idx, selected_features=[5, 6, 7, 8, 9],
            imput_feature='AVG_DP_TUBING', target=1,
           gamma=4, cost=4)

next_sheet = rawdata.parse(sheet_name=sheet_name[1])
class_idx = np.array(next_sheet['class'].values, dtype=np.int16)

SVR_imput(dataset, class_idx, selected_features=[5, 6, 7, 8, 9],
          imput_feature='AVG_DOWNHOLE_TEMPERATURE', target=2,
          gamma=5, cost=3)

next_sheet = rawdata.parse(sheet_name=sheet_name[2])
class_idx = np.array(next_sheet['class'].values, dtype=np.int16)

SVR_imput(dataset, class_idx, selected_features=[5, 6, 7, 8, 9],
          imput_feature='AVG_DOWNHOLE_PRESSURE', target=3,
          gamma=5, cost=5)

next_sheet = rawdata.parse(sheet_name=sheet_name[3])
class_idx = np.array(next_sheet['class'].values, dtype=np.int16)

SVR_imput(dataset, class_idx, selected_features=[1, 2, 3, 5, 6, 7, 8, 9],
          imput_feature='ANNULUS_PRESS', target=4,
          gamma=4, cost=16)

next_sheet = rawdata.parse(sheet_name=sheet_name[4])
class_idx = np.array(next_sheet['class'].values, dtype=np.int16)
SVR_imput(dataset, class_idx, selected_features=[1, 2, 3, 4, 5, 6, 7, 8, 9],
          imput_feature='BORE_OIL_VOL', target=10,
          gamma=0.25, cost=16, run_train=False)

well_code_col = sheet['WELL_BORE_CODE'].values
uniq_well_code = np.unique(well_code_col)

# Normalization
mean, std = np.mean(dataset, axis=0), np.std(dataset, axis=0)
norm_set = (dataset - mean) / (std + 1e-6)

rnn_sets = []
for id, well in enumerate(uniq_well_code):
  rnn_sets.append(norm_set[well_code_col == well, :])

# split a multi-variate sequence into samples
def split_sequence(sequence, n_steps):
  X, y = list(), list()
  for i in range(len(sequence)):
      # find the end of this pattern
      end_ix = i + n_steps
      # check if we are beyond the sequence
      if end_ix > len(sequence):
          break
      # gather input and output parts of the pattern
      seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
      X.append(seq_x)
      y.append(seq_y)
  return np.array(X), np.array(y)

from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import backend as K

def r2(label, pred):
  SS_res = K.sum(K.square(label - pred ))
  SS_tot = K.sum(K.square(label - K.mean(label)))
  return (1 - SS_res/ (SS_tot + K.epsilon()))

# define model
def get_model(n_steps):
  model = Sequential()
  model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, num_features - 1)))
  model.add(LSTM(50, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse', metrics=[r2])
  return model

for n_steps in np.arange(2,18,2):

  train_X = np.empty((0, n_steps, num_features-1))
  train_y = np.empty((0))

  for sequence in rnn_sets:
    X, y = split_sequence(sequence=sequence, n_steps=n_steps)
    train_X = np.append(train_X, np.array(X), axis=0)
    train_y = np.append(train_y, np.array(y), axis=0)


  train_X, train_y = shuffle(train_X, train_y)

  train_X, train_y, test_X, test_y = \
    train_X[int(0.2*data_size):, :], train_y[int(0.2*data_size):], \
    train_X[:int(0.2*data_size),:], train_y[:int(0.2*data_size)]

  folds = list(KFold(n_splits=10, shuffle=False, random_state=None).split(train_X, train_y))

  testScore = []

  for j, (train_idx, val_idx) in enumerate(folds):
    cur_train_X, cur_train_y = train_X[train_idx, :], train_y[train_idx]
    cur_valid_X, cur_valid_y = train_X[val_idx, :], train_y[val_idx]
    model = get_model(n_steps)
    res = model.fit(cur_train_X, cur_train_y, validation_split=0.01, epochs=60, verbose=0)
    _, Score = model.evaluate(cur_valid_X, cur_valid_y, verbose=0)
    testScore.append(Score)

  print('\n', 'n_steps = ', n_steps, 'r2 score = ', np.mean(testScore))


