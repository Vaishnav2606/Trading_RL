from keras.models import Sequential
from keras.layers import LSTM, Dense, CuDNNLSTM
import random
import numpy as np

class Actor:

    def __init__(self,  env, batch_size, learner):
        self.env = env
        self.state = np.array(self.env.reset())
        self.learner = learner
        self.model = self.build_model()
        self.discount_factor = 0.92
        self.local_memory = []
        self.replay_memory = learner.replay_memory
        self.batch_size = batch_size

    def env_reset(self):
        self.state = (self.env.reset())
        


    def build_model(self):
        model = Sequential()
        model.add(CuDNNLSTM(300, input_shape=(self.state.shape[0], self.state.shape[1]), return_sequences=True))
        model.add(CuDNNLSTM(200))
        model.add(Dense(self.learner.action_shape, activation='softmax'))
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model
    
    def update_weights(self):
        self.model.set_weights(self.learner.model.get_weights())
    
    def get_action(self, state):
        state = np.array(state).reshape(self.learner.state_shape)
        # print(state.shape)
        action = np.argmax(self.model.predict(state)[0])
        return action
    
    def compute_TD(self, minibatch):
        curr_state = []
        next_state = []
        actions = []
        rewards = []
        done = []
        qValues = []

        for i in range(self.batch_size):
            curr_state.append(minibatch[i][0].reshape(self.learner.state_shape))
            actions.append(minibatch[i][1])
            rewards.append(minibatch[i][2])
            next_state.append(minibatch[i][3].reshape(self.learner.state_shape))
            done.append(minibatch[i][4])
            qValues.append(self.model.predict(curr_state[i])[0])
            qValue_ns = np.amax(self.model.predict(next_state[i])[0])
            if done[i]:
                qValues[i][actions[i]] = rewards[i]
            else:
                qValues[i][actions[i]] = rewards[i] + self.discount_factor*qValue_ns
        
        qValues = np.array(qValues)
        curr_state = np.array(curr_state)
        actions = np.array(actions)

        pred_qValues = np.array([self.model.predict(x)[0] for x in curr_state])
        indices = np.arange(self.batch_size)
        abs_error = np.abs(qValues[indices, actions]-pred_qValues[indices, actions])

        return abs_error
        

    def perform_action(self):
        action = self.get_action(self.state)
        next_state, reward, done, info = self.env.step(action)
        next_state = np.array(next_state)
        self.local_memory.append([self.state, action, reward, next_state, done])
        if len(self.local_memory) >= self.batch_size:
            minibatch = self.local_memory[-self.batch_size:]
            TD = self.compute_TD(minibatch)
            TD += self.replay_memory.e
            clipped_error = np.minimum(TD, self.replay_memory.absolute_error_upper)
            priority = np.power(clipped_error, self.replay_memory.a)
            for data, p in zip(minibatch, clipped_error):
                self.replay_memory.tree.add(p, data)
                self.replay_memory.memories_n+=1
        self.state = next_state


        