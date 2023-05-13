import gym_anytrading
from gym_anytrading.envs import StocksEnv
import numpy as np
from learner import Learner
import os.path
from yahoofinancials import YahooFinancials
from finta import TA
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date
from datetime import timedelta

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import random


n = int(input('Enter the episode number for the model'))

while os.path.isfile('models/episode_{}.h5'.format(n)) is False:
    print('episode-{} is not available'.format(n))
    n = int(input('Enter the episode number for the model'))

#getting data
symbol = 'TSLA'

years = 2

years = 365*years

endDate = date.today()

startDate = endDate - timedelta(years)

stock = YahooFinancials(symbol)

stock_price_data = pd.DataFrame(stock.get_historical_price_data(str(startDate), str(endDate), 'daily')[symbol]['prices'])

#preprocessing
stock_price_data.drop(columns=['date'], inplace=True)
stock_price_data.dropna(inplace=True)
stock_price_data.rename(columns={'formatted_date':'date'}, inplace=True)
stock_price_data['date'] = pd.to_datetime(stock_price_data['date'])
stock_price_data.set_index('date', inplace=True)
stock_price_data.columns = [x.capitalize() for x in stock_price_data.columns]
stock_ind_data = stock_price_data.copy()

stock_ind_data['RSI'] = TA.RSI(stock_ind_data)
stock_ind_data['MACD'] = TA.MACD(stock_ind_data)['MACD']
stock_ind_data['MACD_SIGNAL'] = TA.MACD(stock_ind_data)['SIGNAL']
stock_ind_data['OBV'] = TA.OBV(stock_ind_data)
stock_ind_data['STOCH'] = TA.STOCH(stock_price_data)
stock_ind_data['STOCHD'] = TA.STOCHD(stock_price_data)
stock_ind_data.dropna(inplace=True)

#Creating Environment
def process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:,'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['RSI', 'MACD', 'MACD_SIGNAL', 'OBV', 'STOCH','STOCHD']]
    return prices, signal_features

class MyStocksEnv(StocksEnv):
    _process_data = process_data


timesteps = 10
test_n = 30
test_env = MyStocksEnv(df=stock_ind_data, frame_bound=(stock_ind_data.shape[0]-test_n, stock_ind_data.shape[0]), window_size=timesteps)

model = load_model('models/episode_{}.h5'.format(n))

state = np.array(test_env.reset())
done = False

state_shape = (1, state.shape[0], state.shape[1])

while not done:
    action = np.argmax(model.predict(state.reshape(state_shape))[0])
    next_state, reward, done, info = test_env.step(action)
    next_state = np.array(next_state)
    if done: 
        print('info: {}'.format(info))
        break
    state = next_state
test_env.render_all()
plt.show()