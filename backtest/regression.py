# -*- coding: utf-8 -*-  
from data import Data
import matplotlib.pyplot as plt
import numpy as np

LINEAR = 0
LOG = 1
EXP = 2

"""
    return newest hold  
"""
def log_func(min_t, max_t):
    return lambda x: 0.9 * np.log(x-min_t + 1) / np.log(max_t-min_t + 1) + 0.1

def exp_func(min_t, max_t):
    return lambda x: 0.9 * (np.exp(x-min_t) - 1) / (np.exp(max_t-min_t) - 1) + 0.1

def linear_func(min_t, max_t):
    return lambda x: 0.9 * (x - min_t) / (max_t - min_t) + 0.1

func_dict = {
        LINEAR : linear_func,
        LOG : log_func,
        EXP : exp_func
        }

default_config = {
        "data_path": "data.csv",
        "min_threshold": 0.000,
        "max_threshold": 0.002,
        "init_money": 1000,
        "func_type": EXP,
        "fee": 0
        }

class Regression:
    def __init__(self, config = default_config):
        self.data = Data(config)
        self.min_t = config["min_threshold"]
        self.max_t = config["max_threshold"]
        self.func_type = config["func_type"]
        self.stock = 0 # num of stocks
        self.fee = config["fee"]
        self.init_money = config["init_money"]
        self.init_state()

    def init_state(self):
        self.step = 0
        self.currency = self.init_money
        self.cur_date = None
        self.cur_price = -1
        self.max_asset = self.init_money
        self.max_drawdown = 0
        self.sharp_rate = 0
        self.action_count = 0
        self.asset_list = [self.init_money, ] # append when sell_out, used for win_rate
        self.asset_list_total = [self.init_money, ] # append when sell_out, used for win_rate
        self.step = 0
        self.has_buy = False
        self.hold = 0 # asset of stock / total asset 


    def do_regression(self, price_list, predict_list, precise_list=None):
        n_day = len(price_list)
        if n_day != len(predict_list):
            raise ValueError("n_days %d not equal!" % len(predict_list))
        self.init_state()
        for price, predict in zip(price_list, predict_list):
            self.tick(price, predict)
        self._sell_out(price_list[n_day-1])

        reward = self.get_reward(n_day)
        win_rate = self.get_win_rate()
        max_drawdown = self.get_max_drawdown()
        win_loss_ratio = self.get_win_loss_ratio()
        return (reward, win_rate, max_drawdown, win_loss_ratio)

    def get_total_asset(self, cur_price):
        """
           total asset
        """
        return self.currency + self.stock * cur_price

    def get_reward(self, n_day, stock_value = 0):
        tmp = (self.currency + stock_value) / self.init_money
        if tmp > 0:
            return (abs(tmp))**(252.0/n_day) - 1
        else:
            return - (abs(tmp))**(252.0/n_day) + 1

    def get_win_rate(self):
        action_count = len(self.asset_list) - 1
        assert(action_count > 0)
        win_count = sum([ 1 if (self.asset_list[i] - self.asset_list[i-1]) >0 else 0 \
                for i in range(1, len(self.asset_list))])
        return win_count*1.0 / action_count

    def get_max_drawdown(self):
        return abs(self.max_drawdown)

    def get_win_loss_ratio(self):
        tmp = [(self.asset_list[i] / self.asset_list[i-1]) - 1 for i in range(1, len(self.asset_list))]
        win = np.sum([item for item in tmp if item > 0])
        loss = - np.sum([item for item in tmp if item < 0])
        return win / loss



    def tick(self, price, predict):
        """
            price : price of i th day  
            predict : predict rate of i+1 th day  
        """
        self.step += 1
        print "tick! %d" % self.step
        # when predict < 0.03  
        # sell out
        f_buy = func_dict[self.func_type](self.min_t, self.max_t)
        cur_asset = self.get_total_asset(price)
        if predict < self.min_t:
            self._sell_out(price)
        elif predict > self.max_t:
        # all buy in 
            amount = self.currency / price
            self._buy(amount, price)
        else:
            hold = f_buy(predict)
            amount = (hold * cur_asset - self.stock * price) / price
            #print "hold = %f, amount = %f" % (hold, amount)
            self._buy(amount, price)
        if abs(cur_asset - self.get_total_asset(price)) > 0.00001:
            print price, predict
            raise ValueError("asset changed! %f != %f" % (cur_asset, self.get_total_asset(price)))
        self.max_asset = max(self.max_asset, cur_asset)
        self.max_drawdown = min(self.max_drawdown, cur_asset / self.max_asset - 1) 
        self.asset_list_total.append(cur_asset)
        


    def _buy(self, amount, price):
        """
           amount : num of stock
           can be positive or negtive
        """
        assert(price > 0)
        #print "ACTION: amount = %f, price = %f" % (amount, price)
        self.stock += amount
        self.currency -= amount * price
        self.hold = self.stock * price / (self.currency + self.stock * price)

    def _sell_out(self, price):
        assert(price > 0)
        if self.stock < 0:
            raise ValueError("stock %f < 0" % self.stock)
        if self.stock > 0:
            print "SELL_OUT: stock = %f, price = %f, currency = %f" % (self.stock, price, self.currency)
            self.asset_list.append(self.get_total_asset(price))
        self.currency += self.stock * price
        self.hold = 0
        self.stock = 0



            
def test(min_t, max_t):
    config = default_config
    config["min_threshold"] = min_t
    config["max_threshold"] = max_t
    re = Regression(config)
    d = Data()
    price_list, predict_list = d.get_data()
    ans = re.do_regression(price_list, predict_list)
    return ans
    
def print_matrix(matrix, x_list, y_list, label = ""):
    print "---------%s----------" % label
    x_len = len(x_list)
    y_len = len(y_list)
    print "\t", 
    for x in x_list:
        print "%.5f\t" % x,
    print 
    for i in range(x_len):
        print "%.4f" % y_list[i],
        for j in range(y_len):
            print "%.4f\t" % matrix[i][j],
        print

def plot_asset(min_t, max_t):
    config = default_config
    config["min_threshold"] = min_t
    config["max_threshold"] = max_t 
    re = Regression(config)
    d = Data()
    price_list, predict_list = d.get_data()
    ans = re.do_regression(price_list, predict_list)
    plt.ylim(1000, 1090)
    plt.plot(re.asset_list_total, color="r")
    plt.title("min_t = %f, max_t %f" % (min_t, max_t))
    #plt.xlabel("平仓次数")
    #plt.ylabel("总资产")
    plt.savefig("asset")

def brute_force():
    min_t_down = -0.005
    min_t_up = 0.0001
    max_t_down = 0.0002
    max_t_up = 0.002
    rewards= []
    win_rates = []
    max_drawdowns = []
    win_loss_ratios = []
    min_t_list = [min_t_down + i * (min_t_up - min_t_down) / 10 for i in range(1,11)]
    max_t_list = [max_t_down + i * (max_t_up - max_t_down) / 10 for i in range(1,11)]
    for min_t in min_t_list:
        tmp_reward = []
        tmp_win_rate = []
        tmp_max_drawdown=[]
        tmp_win_loss_ratio=[]
        for max_t in max_t_list:
            print "min_t = %f, max_t = %f" % (min_t, max_t) 
            ans = test(min_t, max_t)
            tmp_reward.append(ans[0])
            tmp_win_rate.append(ans[1])
            tmp_max_drawdown.append(ans[2])
            tmp_win_loss_ratio.append(ans[3])
        rewards.append(tmp_reward)
        win_rates.append(tmp_win_rate)
        max_drawdowns.append(tmp_max_drawdown)
        win_loss_ratios.append(tmp_win_loss_ratio)
    print_matrix(rewards, max_t_list, min_t_list, label = "reward")
    print_matrix(win_rates, max_t_list, min_t_list, label = "win_rate")
    print_matrix(max_drawdowns, max_t_list, min_t_list, label = "max_drawdown")
    print_matrix(win_loss_ratios, max_t_list, min_t_list, label = "win_loss_ratio")

if __name__=="__main__":
    #test(-0.003, 0.0002)
    brute_force()
    #plot_asset(-0.003, 0.0002) # high risk
    #plot_asset(0.0001, 0.00018) # mid risk
    #plot_asset(0, 0.0002) # low risk






        


