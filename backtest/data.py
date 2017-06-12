import pandas as pd

default_config = {
        "data_path": "predict_train.csv",
        }

class Data:
    def __init__(self, config = default_config):
        self.df = pd.read_csv(config["data_path"])

    def get_data(self):
        return self.df["price"], self.df["predict"] 

if __name__=="__main__":
    import matplotlib.pyplot as plt
    d = Data(default_config)
    price_list, predict_list = d.get_data()
    predict_list.hist()
    plt.savefig("lala")
    
    


