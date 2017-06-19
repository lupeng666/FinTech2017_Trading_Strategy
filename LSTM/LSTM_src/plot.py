import matplotlib.pyplot as plt


def plot_results(result_df):
    fig = plt.figure(facecolor='white')
    plt.scatter(result_df['label'], result_df['pred'])
    plt.xlabel('label')
    plt.ylabel('pred')
    plt.grid()
    plt.savefig('..//LSTM_output//scatter_result.png')
    plt.show()


def plot_accumulate_precision(result_df):
    result_df['pred_bool'] = result_df['pred'] * result_df['label'] > 0
    result_df['flag'] = range(1, len(result_df) + 1)
    result_df['accumulate_precision'] = result_df['pred_bool'].cumsum() / result_df['flag']

    print("total precision:", result_df['accumulate_precision'].values[-1])

    fig = plt.figure(facecolor='white')
    plt.plot(result_df['accumulate_precision'], label='Prediction')
    plt.xlabel('sample_num')
    plt.ylabel('accumulate_precision')

    plt.grid()
    plt.savefig('..//LSTM_output//accumulate_precision.png')
    plt.show()
