import lstm
import time
import plot
from keras.models import save_model, load_model
from keras.utils.vis_utils import plot_model

# Main Run Thread
if __name__ == '__main__':
    global_start_time = time.time()
    epochs = 80
    seq_len = 30

    print('> Loading features... ')
    X_train, absolute_price_train, y_train, X_test, absolute_price_test, y_test = \
        lstm.load_feat('..//LSTM_data//price.csv', '..//LSTM_data//refer.csv', seq_len)

    print('> features Loaded. Compiling...')
    model = lstm.build_model()

    plot_model(model, "..//LSTM_output//model.png")

    model.fit(
        [X_train, absolute_price_train],
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.2)
    save_model(model, "..//LSTM_output//model.hdf5")

    predictions = lstm.predict_and_postprocess(model, [X_test, absolute_price_test])

    print('Training duration (s) : ', time.time() - global_start_time)
    plot.plot_results(predictions)
    plot.plot_accumulate_precision(predictions)
