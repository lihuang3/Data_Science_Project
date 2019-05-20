import sys, os, math
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, KFold
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np, pandas

dir =  os.getcwd()
# fix random seed for reproducibility
np.random.seed(7)

rawdata = pandas.ExcelFile(os.path.join(dir,'data','datasets.xlsx'))
well_dict = {7405: '1C', 7078: '11H', 5599: '12H', 5351: '14H', 7289: '15D', 5693: '4AH', 5769: '5AH'}

# Convert datetime to integer
label = 'DATEPRD'
sheet_name = rawdata.sheet_names
sheet = rawdata.parse(sheet_name=sheet_name[0])
rawdates = pandas.to_datetime(sheet[label].values)
sheet[label].values.dtype = np.int64
for id, it in enumerate(rawdates):
    sheet[label].values[id] = 10000*rawdates[id].year + 100*rawdates[id].month + rawdates[id].day

# Configure relevant features for data imputation and oil productin prediction
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

# Def r2 metric in keras
def r2metric(label, pred):
    SS_res = K.sum(K.square(label - pred ))
    SS_tot = K.sum(K.square(label - K.mean(label)))
    return (1 - SS_res/ (SS_tot + K.epsilon()))

# Def r2 metric for general usage
def r2m(label, pred):
    SS_res = np.sum(np.square(label - pred ))
    SS_tot = np.sum(np.square(label - np.mean(label)))
    return (1 - SS_res/ (SS_tot + 1e-9))

# Def mlp model
def get_model(layers, input_dim):
    """
    :param layers: list of integers
        a list specifiy number of neurons in each layer, e.g. layers=[64, 32]
    :param input_dim: integer
        the number of input variables
    :return:
        an mlp model
    """
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_dim=input_dim))
    for layer in layers[1:]:
        model.add(Dense(layer, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=[r2metric])
        return model

# Def mlp imputation function
def MLP_imput(dataset, class_idx, selected_features, imput_feature,
              target, layers, epochs=100,
              cross_val=False, nfolds=10, run_train=False):

    """
    :param dataset: numpy nd array
        an m * n numpy array, with m the len of data, n the num of variables
    :param class_idx: integer
        the label for train-test split, refering the last column of each sheet,
        0: missing data; 1-2: test set data; 3-10: training set data
    :param selected_features: list of integers
        an array of indices for refering to 'relevant_features'
    :param imput_feature: string
        the imputing feature name (def in 'relevant_features')
    :param target: integer
        the imputing feature index (def in 'relebant_features')
    :param layers: list of integers
        a list specifiy number of neurons in each layer, e.g. layers=[64, 32]
    :param epochs: intger
        number of epochs for mlp training
    :param cross_val: bool
        if True, execute cross validation
    :param nfolds: integer
        nfolds for cross validation
    :param run_train: bool
        if True, retrain the model; otherwise load a saved model
    :return:
    """
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

    # train-test split
    train_X, test_X, train_y, test_y = \
    train_set[:, selected_features], test_set[:, selected_features], \
    train_set[:, target], test_set[:, target]
    for _ in range(5):
        train_X, train_y = shuffle(train_X, train_y)
    pred_X, pred_y = missing_set[:, selected_features], missing_set[:, target]

    if not run_train and os.path.isfile( os.path.join(dir, 'models', '%s_model.json'%(imput_feature))):
        f = open(os.path.join(dir, 'models', '%s_model.json'%(imput_feature)),'r')
        loaded_json = f.read()
        f.close()
        model = model_from_json(loaded_json)
        model.load_weights(os.path.join(dir, 'models', '%s_model.h5'%(imput_feature)))
        model.compile(optimizer='adam', loss='mse', metrics=[r2metric])
        print("Model loaded from %s_model.h5"%(imput_feature))
    else:
        if cross_val:
            X, y = np.copy(train_X), np.copy(train_y)
            folds = list(KFold(n_splits=nfolds, shuffle=False, random_state=None).split(X, y))
            r2s, rmses =[], []
            for j, (_train_idx, _val_idx) in enumerate(folds):
                _train_X, _train_y = X[_train_idx, :], y[_train_idx]
                _valid_X, _valid_y = X[_val_idx, :], y[_val_idx]
                model = get_model(layers, num_features)
                model.fit(_train_X, _train_y, epochs=epochs, verbose=0)
                mse, r2 = model.evaluate(_valid_X, _valid_y, verbose=0)
                print('Fold ',j, '\n', 'rmse = ', np.sqrt(mse), '\n', 'r2 score = ', r2)
                r2s.append(r2)
                rmses.append(np.sqrt(mse))
            print('\n', 'cv r2 = ', np.mean(r2s), '\n', 'cv rmse = ', np.mean(rmses))

        model = get_model(layers, num_features)
        res = model.fit(train_X, train_y, epochs=epochs, verbose=0)
        print('\n', 'train r2 = ', res.history['r2metric'][-1], '\n', 'train rmse = ', np.sqrt(res.history['loss'][-1]))

        model_json = model.to_json()
        with open(os.path.join(dir, 'models', "%s_model.json"%(imput_feature)), "w") as f:
          f.write(model_json)
        model.save_weights(os.path.join(dir, 'models', "%s_model.h5"%(imput_feature)))
        print("Saving %s model to disk .."%(imput_feature))

    pred_y = model.predict(test_X)
    r2 = r2m(label=test_y, pred=np.squeeze(pred_y))
    mse, _ = model.evaluate(test_X, test_y, verbose=0)
    print('\n', 'test r2 = ', r2, '\n', 'test rmse = ', np.sqrt(mse))

    if not imput_feature == 'BORE_OIL_VOL':
        pred_y = model.predict(pred_X) * std[target] + mean[target]
        pred_y = np.squeeze(pred_y)
        dataset[class_idx==0, target] = np.copy(pred_y)

class_idx = np.array(sheet['class'].values, dtype=np.int16)

# Imput missing data in AVG_DP_TUBING
MLP_imput(dataset, class_idx, selected_features=[5, 6, 7, 8, 9],
            imput_feature='AVG_DP_TUBING', target=1,
            layers=[32,16], epochs=100, cross_val=False, run_train=False)

# Imput missing data in AVG_DOWNHOLE_TEMPERATURE
next_sheet = rawdata.parse(sheet_name=sheet_name[1])
class_idx = np.array(next_sheet['class'].values, dtype=np.int16)
MLP_imput(dataset, class_idx, selected_features=[5, 6, 7, 8, 9],
            imput_feature='AVG_DOWNHOLE_TEMPERATURE', target=2,
            layers=[32,16], cross_val=False, run_train=False)

# Imput missing data in AVG_DOWNHOLE_PRESSURE
next_sheet = rawdata.parse(sheet_name=sheet_name[2])
class_idx = np.array(next_sheet['class'].values, dtype=np.int16)
MLP_imput(dataset, class_idx, selected_features=[5, 6, 7, 8, 9],
            imput_feature='AVG_DOWNHOLE_PRESSURE', target=3,
            layers=[32,16], cross_val=False, run_train=False)

# Imput missing data in ANNULUS_PRESS
next_sheet = rawdata.parse(sheet_name=sheet_name[3])
class_idx = np.array(next_sheet['class'].values, dtype=np.int16)
MLP_imput(dataset, class_idx, selected_features=[1, 2, 3, 5, 6, 7, 8, 9],
            imput_feature='ANNULUS_PRESS', target=4,
            layers=[64,32], epochs=200, cross_val=False, run_train=False)

# predict on oil production
next_sheet = rawdata.parse(sheet_name=sheet_name[4])
class_idx = np.array(next_sheet['class'].values, dtype=np.int16)
MLP_imput(dataset, class_idx, selected_features=[1, 2, 3, 4, 5, 6, 7, 8, 9],
          imput_feature='BORE_OIL_VOL', target=10,
          layers=[32,16], epochs=150, cross_val=False, run_train=True)


