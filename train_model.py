import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from yahoofinancials import YahooFinancials
from finta import TA
import gym_anytrading
from gym_anytrading.envs import StocksEnv

from datetime import date
from datetime import timedelta

from keras.models import Sequential
from keras.layers import LSTM, Dense
import random

from memory import Memory
from actor import Actor
from learner import Learner



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

def build_actors(df, batch_size, learner, test_n, n):
    actors = []
    for i in range(n):
        env = MyStocksEnv(df=stock_ind_data, frame_bound=(batch_size, stock_ind_data.shape[0]-test_n), window_size=batch_size)
        actors.append(Actor(env, batch_size=batch_size, learner=learner))
    return actors

timesteps = 10
test_n = 30
env = MyStocksEnv(df=stock_ind_data, frame_bound=(timesteps, stock_ind_data.shape[0]-test_n), window_size=timesteps)

#training
episodes = 10
state = np.array(env.reset())
learner = Learner(state_shape = state.shape, action_shape = env.action_space.n, memory_size=100, ddqn_flag=True)
actors_n = 4
actors = build_actors(df=stock_ind_data, batch_size=timesteps, learner=learner, n=actors_n, test_n=test_n)

infos = {}

for e in range(episodes):
    done = False
    r = 0
    state = np.array(env.reset())
    while not done:
        for actor in actors:
            actor.perform_action()
        action = learner.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.array(next_state)
        learner.experience_replay()
        for actor in actors:
            actor.update_weights()
        if done:
            for actor in actors:
                actor.env_reset()
            infos[e]=info
            learner.model.save('models/episode_{}.h5'.format(e))
            print("Episode: {}, Info: ()".format(e, info))
            break
        state = next_state
        
print(pd.DataFrame(infos))
    
            
