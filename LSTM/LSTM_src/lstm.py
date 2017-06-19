import os
import time
import warnings
import numpy as np
from keras.layers import Input, merge, LSTM, Dense
from keras.models import Model
from tensorflow import atan
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


def normalize(window_data):
    normalised_data = []
    for df in window_data:
        for i in np.arange(len(df) - 1, -1, -1):
            df[i] = df[i] / df[0] - 1
        normalised_data.append(df.copy())
        del df
    return normalised_data


def load_feat(price_filename, index_filename, seq_len, train_test_split=0.9, normalise_window=True):
    price_df = pd.read_csv(price_filename, index_col='Time', parse_dates=True, encoding='gb2312')
    index_df = pd.read_csv(index_filename, index_col='Time', parse_dates=True, encoding='gb2312')
    index_df = index_df[['TIPS-5Y', 'TIPS-10Y', 'TIPS-20Y', 'TIPS-LONG',
                         'UST BILL 10-Y RETURN', 'LIBOR-OVERNIGHT', 'SPDR:t', 'USD/CNY']]
    df = pd.merge(price_df, index_df, left_index=True, right_index=True, how='inner')
    df = df[:'2016-12-08']

    absolute_price = df.values[seq_len:-1, 0]

    # normalize each feature
    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) + 0.001)
    df = df.values

    sequence_length = seq_len + 1
    result = []
    for index in range(len(df) - sequence_length):
        result.append(df[index: index + sequence_length].copy())

    if normalise_window:
        result = normalize(result)
        absolute_price = np.around((absolute_price - 1000) / 1000, decimals=1)

    result = np.array(result)

    row = round(train_test_split * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)

    x_train = train[:, :-1, :]
    y_train = train[:, -1, 0]

    absolute_price_train = absolute_price[:int(row)]
    absolute_price_test = absolute_price[int(row):]

    x_test = result[int(row):, :-1, :]
    y_test = result[int(row):, -1, 0]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_train.shape[2]))

    return [x_train, absolute_price_train, y_train, x_test, absolute_price_test, y_test]


def build_model():
    lstm_input_1 = Input(shape=(30, 9), name='lstm_input')
    lstm_output_1 = LSTM(128, activation=atan, return_sequences=True, dropout_W=0.2, dropout_U=0.1)(lstm_input_1)
    lstm_output_2 = LSTM(256, activation=atan, dropout_W=0.2, dropout_U=0.1)(lstm_output_1)
    aux_input = Input(shape=(1,), name='aux_input')
    merged_data = merge([lstm_output_2, aux_input], mode='concat', concat_axis=-1)
    Dense_output_1 = Dense(64, activation='linear')(merged_data)
    Dense_output_2 = Dense(16, activation='linear')(Dense_output_1)
    predictions = Dense(1, activation='linear')(Dense_output_2)

    model = Model(input=[lstm_input_1, aux_input], output=predictions)

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

    return model


def predict_and_postprocess(model, data_list):
    data, absolute_price = data_list
    predicted = model.predict(data_list)

    last_day = data[:, -1, 0]

    predicted = np.reshape(predicted, (predicted.size,))
    last_day = np.reshape(last_day, (last_day.size,))

    ratio = (predicted - last_day) / (last_day + 1)
    tmp_df = pd.DataFrame({'pred': ratio})

    price_df = pd.read_csv('..//LSTM_data//price.csv', index_col='Time', parse_dates=True, encoding='gb2312')
    price_df = price_df[:'2016-12-08']
    price_df['delta'] = price_df.price.diff().shift(-1)
    price_df['ratio'] = price_df['delta'] / price_df['price']
    tmp_df['label'] = price_df['ratio'][-len(tmp_df) - 1:-1].values
    tmp_df['price'] = price_df['price'][-len(tmp_df) - 1:-1].values
    tmp_df.index = price_df['price'][-len(tmp_df) - 1:-1].index
    tmp_df.to_csv('..//LSTM_output//pred_result.csv')

    return tmp_df[['pred', 'label']]
