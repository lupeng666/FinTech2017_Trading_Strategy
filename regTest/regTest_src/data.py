import pandas as pd
import matplotlib.pyplot as plt
import os

default_config = {
    "data_path": "../regTest_data/predict.csv",
}


class Data:
    def __init__(self, config=default_config):
        self.df = pd.read_csv(config["data_path"])

    def get_data(self):
        return self.df["price"], self.df["predict"]


def plot_hist(output="predict_hist"):
    """
        plot distribution (value count) of predict
        and save as $output
    """
    d = Data(default_config)
    finput = os.path.split(default_config["data_path"])[-1]
    price_list, predict_list = d.get_data()
    predict_list.hist()
    plt.title("input_file: %s" % finput)
    plt.xlabel("predict value")
    plt.ylabel("count")
    foutput = os.path.join("..", "regTest_output", output)
    plt.savefig(output)


if __name__ == "__main__":
    plot_hist("predict_hist")
