import sys, os, math
dir =  os.getcwd()
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np, pandas
from sklearn.model_selection import KFold
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dropout
from keras.layers import Dense
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(7)

rawdata = pandas.ExcelFile(os.path.join(dir,'data','datasets_wi.xlsx'))

vars_list = ['BORE_OIL_VOL','ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
             'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHT_P',
             'AVG_WHP_P', 'DP_CHOKE_SIZE', 'BORE_GAS_VOL', 'BORE_WAT_VOL','BORE_WI_VOL', 'BORE_WI_VOL','DATEPRD']

label = 'DATEPRD'
sheet_name = rawdata.sheet_names
sheet = rawdata.parse(sheet_name=sheet_name[0])
rawdates = pandas.to_datetime(sheet[label].values)

for id, it in enumerate(rawdates):
  sheet[label].values[id] = 10000*rawdates.year[id] + 100*rawdates.month[id] + rawdates.day[id]

dates = np.array(sheet[label].values, dtype=np.int64)
uniq_t = np.sort(np.unique(dates))
t_dict = {i:j for i,j in zip(uniq_t, np.arange(np.shape(uniq_t)[0]) )}

irrelevant_features = ['DATEPRD', 'WELL_BORE_CODE',
                       'FLOW_KIND', 'BORE_GAS_VOL',
                       'BORE_WAT_VOL']

relevant_features = ['AVG_DP_TUBING', 'AVG_DOWNHOLE_TEMPERATURE',
                     'AVG_DOWNHOLE_PRESSURE', 'AVG_ANNULUS_PRESS',
                     'ON_STREAM_HRS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P',
                     'AVG_WHT_P', 'DP_CHOKE_SIZE', 'WI1', 'WI2',
                     'BORE_OIL_VOL']

num_features = len(relevant_features)
data_size = len(sheet[relevant_features[0]].values)
dataset = np.empty((data_size, num_features))
for idx, feature in enumerate(relevant_features):
  dataset[:, idx] = np.array(sheet[feature].values, dtype=np.float32)

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

def r2metric(label, pred):
  SS_res = K.sum(K.square(label - pred ))
  SS_tot = K.sum(K.square(label - K.mean(label)))
  return (1 - SS_res/ (SS_tot + K.epsilon()))

# define model
def get_model(n_steps, neurons):
  model = Sequential()
  model.add(LSTM(neurons, activation='tanh', input_shape=(n_steps, num_features - 1)))
  # model.add(LSTM(50, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse', metrics=[r2metric])
  return model

def load_model(model_name):
  # Load model
  f = open(os.path.join(dir, 'models', '%s_model.json' % (model_name)), 'r')
  loaded_json = f.read()
  f.close()
  model = model_from_json(loaded_json)
  model.load_weights(os.path.join(dir, 'models', '%s_model.h5' % (model_name)))
  # print("Model loaded from %s_model.h5" % (model_name))
  model.compile(optimizer='adam', loss='mse', metrics=[r2metric])
  return model

def save_model(model, model_name):
  model_json = model.to_json()
  with open(os.path.join(dir, 'models', "%s_model.json" % (model_name)), "w") as f:
    f.write(model_json)
  model.save_weights(os.path.join(dir, 'models', "%s_model.h5" % (model_name)))
  # print("Saving %s model to disk .." % (model_name))

def train_test_split(sequence, n_steps, num_features, norm=True):

  train_X = np.empty((0, n_steps, num_features-1))
  train_y = np.empty((0))

  test_X = np.empty((0, n_steps, num_features-1))
  test_y = np.empty((0))

  X, y = split_sequence(sequence=sequence, n_steps=n_steps)
  X, y = shuffle(X, y)

  data_size = np.shape(sequence)[0]
  train_X, test_X = X[int(0.2*data_size):, :], X[:int(0.2*data_size), :]
  train_y, test_y = y[int(0.2*data_size):], y[:int(0.2*data_size)]

  if norm:
    Xmean, Xstd = np.mean(train_X, axis=0), np.std(train_X, axis=0)
    ymean, ystd = np.mean(train_y, axis=0), np.std(train_y, axis=0)
    train_X = (train_X - Xmean) / Xstd
    train_y = (train_y - ymean) / ystd
    test_X = (test_X - Xmean) / Xstd
    test_y = (test_y - ymean) / ystd
  else:
    Xmean, Xstd, ymean, ystd = None, None, None, None

  return train_X, train_y, test_X, test_y, Xmean, Xstd, ymean, ystd

def lstm_model(model_name, n_steps, X, y, neurons=64, epochs=50, cross_val=False, nfolds=10, run_train=False):

  if not run_train and os.path.isfile(os.path.join(dir, 'models', '%s_model.json' % (model_name))):
    model = load_model(model_name)
  else:
    if cross_val:
      folds = list(KFold(n_splits=nfolds, shuffle=False, random_state=None).split(X, y))
      r2s, mses =[], []
      for j, (train_idx, val_idx) in enumerate(folds):
        train_X, train_y = X[train_idx, :], y[train_idx]
        valid_X, valid_y = X[val_idx, :], y[val_idx]
        model = get_model(n_steps, neurons=neurons)
        model.fit(train_X, train_y, validation_split=0.01, epochs=epochs, verbose=0)
        mse, r2 = model.evaluate(valid_X, valid_y, verbose=0)
        print('Fold ',j, '\n', 'mse = ', mse, '\n', 'r2 score = ', r2)
        r2s.append(r2)
        mses.append(mses)
      print('\n', 'n_steps = ', n_steps, 'cv r2 = ', np.mean(r2s), '\n', 'cv mse = ', np.mean(mses))

    model = get_model(n_steps, neurons=neurons)
    model.fit(X, y, epochs=epochs, verbose=0)
    save_model(model, model_name)

  return model

# Finetune on each well
def model_finetune(new_model, default_model, X, y, epochs=10, cross_val=True, nfolds=10, run_train=True):

  if not run_train and os.path.isfile(os.path.join(dir, 'models', '%s_model.json' % (new_model))):
    model = load_model(new_model)
  else:
    if cross_val:
      folds = list(KFold(n_splits=nfolds, shuffle=False, random_state=None).split(X, y))
      mses, r2s = [], []
      for j, (train_idx, val_idx) in enumerate(folds):
        model = load_model(default_model)
        train_X, train_y = X[train_idx, :], y[train_idx]
        valid_X, valid_y = X[val_idx, :], y[val_idx]
        model.fit(train_X, train_y, epochs=epochs, verbose=0)
        mse, r2 = model.evaluate(valid_X, valid_y, verbose=0)
        print('Fold ',j, '\n', 'mse = ', mse, '\n', 'r2 score = ', r2)
        r2s.append(r2)
        mses.append(mses)

      print('\n', 'cv r2 = ', np.mean(r2s), '\n', 'cv mse = ', np.mean(mses))

    model = load_model(default_model)
    model.fit(X, y, epochs=epochs, verbose=0)
    save_model(model, new_model)

  return model


well_code_col = sheet['WELL_BORE_CODE'].values
uniq_well_code = np.unique(well_code_col)
well_dict = {}
for well in uniq_well_code:
  wellname = well.replace("/","")
  wellname = wellname.replace("-","")
  wellname = wellname.replace(" ","")
  well_dict[well] =  wellname[wellname.find('F')+1:]

n_steps = 10

"""2
Method 1: norm all wells
"""
def method1():

  train_X, train_y, test_X, test_y, Xmean, Xstd, ymean, ystd = train_test_split(dataset, n_steps, num_features)
  # set run_train = True to train the model
  # set cross_val = True to enable cross validation
  model = lstm_model(model_name='lstm10wi', X=train_X, y=train_y,
                     n_steps=n_steps, cross_val=False, run_train=False)
  mse, r2 = model.evaluate(test_X, test_y, verbose=0)
  print('All wells\n','mse = ', mse, '\n', 'r2 = ', r2)


"""
Norm all wells, train-test-split by each well
"""
def method2():
  mses, r2s, metric2s = [], [], []
  test_size = 0

  train_X = np.empty((0, n_steps, num_features - 1))
  train_y = np.empty((0))
  sep_train_X, sep_train_y = [], []

  test_X = np.empty((0, n_steps, num_features - 1))
  test_y = np.empty((0))
  sep_test_X, sep_test_y = [], []

  sep_well_Xmean, sep_well_Xstd, sep_well_ymean, sep_well_ystd = [], [], [], []

  for id, well in enumerate(uniq_well_code):
    wellname = well_dict[well]
    datesbywell = np.array(sheet['DATEPRD'].values[well_code_col == well], dtype=np.int64)
    # make sure sequences are sorted by dates
    subdataset = dataset[well_code_col==well, :]
    data_size = np.shape(subdataset)[0]
    subdataset = subdataset[datesbywell.argsort()]
    output = train_test_split(subdataset, n_steps, num_features, norm=False)
    train_X = np.append(train_X, output[0], axis=0)
    train_y = np.append(train_y, output[1], axis=0)
    test_X = np.append(test_X, output[2], axis=0)
    test_y = np.append(test_y, output[3], axis=0)

    sep_train_X.append(output[0])
    sep_train_y.append(output[1])
    sep_test_X.append(output[2])
    sep_test_y.append(output[3])
    sep_well_Xmean.append(output[4])
    sep_well_Xstd.append(output[5])
    sep_well_ymean.append(output[6])
    sep_well_ystd.append(output[7])

  train_X, train_y = shuffle(train_X, train_y)
  Xmean, Xstd = np.mean(train_X, axis=0), np.std(train_X, axis=0)
  ymean, ystd = np.mean(train_y, axis=0), np.std(train_y, axis=0)

  train_X = (train_X - Xmean) / Xstd
  train_y = (train_y - ymean) / ystd
  test_X = (test_X - Xmean) / Xstd
  test_y = (test_y - ymean) / ystd

  model0 = lstm_model('lstm10wi0', n_steps=n_steps,
                      X=train_X, y=train_y, cross_val=False, run_train=False)


  mse, r2 = model0.evaluate(test_X, test_y, verbose=0)
  pred_y = model0.predict(test_X)
  metric2 = np.mean(np.abs(np.reshape(pred_y, [-1, 1]) - np.reshape(test_y, [-1, 1])) / (test_y + ymean / ystd))
  print('mse = %.3f '%(mse) , 'r2 = %.3f'%(r2), 'metric2 = %.2f'%(metric2))


  for id, well in enumerate(uniq_well_code):
    wellname = well_dict[well]
    print('\n', wellname)
    test_X, test_y = sep_test_X[id], sep_test_y[id]
    test_X = (test_X - Xmean) / Xstd
    test_y = (test_y - ymean) / ystd
    mse, r2 = model0.evaluate(test_X, test_y, verbose=0)
    pred_y = model0.predict(test_X)

    metric2 = np.mean(np.abs(np.reshape(pred_y, [-1, 1]) - np.reshape(test_y, [-1, 1] )) / (test_y + ymean/ystd))
    print('mse = %.3f '%(mse) , 'r2 = %.3f'%(r2), 'metric2 = %.2f'%(metric2))



"""
Norm by each well, train-test split by each well
"""
def method3():
  mses, r2s, metric2s = [], [], []
  test_size = 0

  train_X = np.empty((0, n_steps, num_features - 1))
  train_y = np.empty((0))
  sep_train_X, sep_train_y = [], []

  test_X = np.empty((0, n_steps, num_features - 1))
  test_y = np.empty((0))
  sep_test_X, sep_test_y = [], []

  sep_well_Xmean, sep_well_Xstd, sep_well_ymean, sep_well_ystd = [], [], [], []
  for id, well in enumerate(uniq_well_code):
    wellname = well_dict[well]
    datesbywell = np.array(sheet['DATEPRD'].values[well_code_col == well], dtype=np.int64)
    # make sure sequences are sorted by dates
    subdataset = dataset[well_code_col==well, :]
    data_size = np.shape(subdataset)[0]
    print('\n', wellname, ' size=', data_size)
    subdataset = subdataset[datesbywell.argsort()]
    output = train_test_split(subdataset, n_steps, num_features)
    train_X = np.append(train_X, output[0], axis=0)
    train_y = np.append(train_y, output[1], axis=0)
    test_X = np.append(test_X, output[2], axis=0)
    test_y = np.append(test_y, output[3], axis=0)

    sep_train_X.append(output[0])
    sep_train_y.append(output[1])
    sep_test_X.append(output[2])
    sep_test_y.append(output[3])
    sep_well_Xmean.append(output[4])
    sep_well_Xstd.append(output[5])
    sep_well_ymean.append(output[6])
    sep_well_ystd.append(output[7])

  train_X, train_y = shuffle(train_X, train_y)
  model1 = lstm_model('lstm10wi1', n_steps=n_steps,
                      X=train_X, y=train_y, cross_val=False, run_train=False)

  # Convert raw data to sequences by wells


  mses, r2s, metric2s = [], [], []
  test_size = 0
  for id, well in enumerate(uniq_well_code):
    wellname = well_dict[well]
    print('\n', wellname)
    train_X, train_y = sep_train_X[id], sep_train_y[id]
    fine_model = model_finetune(new_model=wellname, default_model='lstm10wi1',
                                X=train_X, y=train_y, epochs=20,
                                cross_val=False, run_train=True)

    test_X, test_y = sep_test_X[id], sep_test_y[id]
    mse, r2 = fine_model.evaluate(test_X, test_y, verbose=0)
    pred_y = fine_model.predict(test_X)
    well_ymean = sep_well_ymean[id]
    well_ystd = sep_well_ystd[id]
    metric2 = np.mean(np.abs(np.reshape(pred_y, [-1, 1]) - np.reshape(test_y, [-1, 1] )) / (test_y + well_ymean[-1]/well_ystd[-1]))
    print('mse = %.3f '%(mse) , 'r2 = %.3f'%(r2), 'metric2 = %.2f'%(metric2))
    mses.append(mse*np.shape(test_y)[0])
    r2s.append(r2*np.shape(test_y)[0])
    metric2s.append(metric2*np.shape(test_y)[0])
    test_size += np.shape(test_y)[0]

  mse = np.sum(mses)/test_size
  r2 = np.sum(r2s)/test_size
  metric2 = np.sum(metric2s)/test_size
  print('mse = %.3f '%(mse) , 'r2 = %.3f'%(r2), 'metric2 = %.2f'%(metric2))


"""
Norm by each well group, train-test split by each well
"""

"""
Method 4
"""
def group_model(modelname, n_steps=7, neurons=64, epochs=80, well_group = None):
  print(well_group)
  print('nstep = ', n_steps, ' epochs = ',epochs)
  train_X = np.empty((0, n_steps, num_features - 1))
  train_y = np.empty((0))
  test_X = np.empty((0, n_steps, num_features - 1))
  test_y = np.empty((0))

  # Group 1, Well 12H + 14H
  individual_norm = False

  for id, well in enumerate(uniq_well_code):
    if well_dict[well] not in well_group:
      continue

    datesbywell = np.array(sheet['DATEPRD'].values[well_code_col == well], dtype=np.int64)
    # make sure sequences are sorted by dates
    subdataset = dataset[well_code_col==well, :]
    data_size = np.shape(subdataset)[0]
    # subdataset = subdataset[datesbywell.argsort()]
    output = train_test_split(subdataset, n_steps, num_features, norm=individual_norm)
    train_X = np.append(train_X, output[0], axis=0)
    train_y = np.append(train_y, output[1], axis=0)
    test_X = np.append(test_X, output[2], axis=0)
    test_y = np.append(test_y, output[3], axis=0)

  if not individual_norm:
    Xmean, Xstd = np.mean(train_X, axis=0), np.std(train_X, axis=0)
    ymean, ystd = np.mean(train_y, axis=0), np.std(train_y, axis=0)
    train_X = (train_X - Xmean) / Xstd
    train_y = (train_y - ymean) / ystd
    test_X = (test_X - Xmean) / Xstd
    test_y = (test_y - ymean) / ystd

  for _ in range(5):
    train_X, train_y = shuffle(train_X, train_y)

  model = lstm_model('lstmG_%s'%(modelname), neurons=neurons, n_steps=n_steps, epochs=epochs,
                      X=train_X, y=train_y, cross_val=False, run_train=True)

  mse, r2 = model.evaluate(test_X, test_y, verbose=0)
  pred_y = model.predict(test_X)
  print('mse = %.3f '%(mse) , 'r2 = %.3f'%(r2))

# well_group = ['12H','14H']
# group_model(modelname='G1',epochs=80, well_group=well_group)
#
# well_group = ['11H', '1C']
# n_steps=9
# epoch=90
# group_model(modelname=str(epoch)+'G3',n_steps=n_steps, neurons=64, epochs=epoch, well_group=well_group)

#
# well_group = ['15D']
# for epoch in [110,110,110,110,110,110]:
#   group_model(modelname='G1_%d'%(epoch),n_steps=9, neurons=64, epochs=epoch, well_group=well_group)
#

