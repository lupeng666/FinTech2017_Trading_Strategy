import lstm
import time
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import save_model, load_model
from keras.utils.vis_utils import plot_model


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


# Main Run Thread
if __name__ == '__main__':
    global_start_time = time.time()
    epochs = 50
    seq_len = 30

    print('> Loading data... ')
    # df = pd.read_csv('price0.csv', index_col='Time', parse_dates=True, encoding='gb2312') #
    X_train, y_train, X_test, y_test = lstm.load_feat('price_one_stock.csv', 'refer.csv', seq_len, True)

    # X_train, y_train, X_test, y_test = lstm.load_data('price.csv', seq_len, True)
    # print(X_train.shape, y_train.shape)
    # pd.DataFrame(y_test).isnull().any()
    pd.DataFrame(X_train[:, 1]).isnull().any()
    X_train[0, 1]
    print('> Data Loaded. Compiling...')

    model = lstm.build_model([4, 50, 100, 1])
    # model = lstm.build_model()

    # plot_model(model, "mse_price_ratio1.png")

    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.2)
    save_model(model, "mse_price_ratio1_0.hdf5")

    # model = load_model("mse_price_ratio1.hdf5")
    # predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
    # predicted = lstm.predict_sequence_full(model, X_test, seq_len)
    predictions = lstm.predict_point_by_point(model, X_test)
    # print(predictions)

    print('Training duration (s) : ', time.time() - global_start_time)
    # plot_results_multiple(predictions, y_test, 50)
    print(len(predictions))
    print(len(y_test))
    pd.DataFrame({'pred': predictions, 'label': y_test}).to_csv('tmp_result1_0.csv')
    plot_results(predictions, y_test)

    # predictions
