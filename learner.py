from keras.models import Sequential
from keras.layers import LSTM, Dense, CuDNNLSTM
import random
import numpy as np
from memory import Memory

class Learner:
    
    def __init__(self, state_shape, action_shape, memory_size, batch_size=24, ddqn_flag=False):
        self.ddqn = ddqn_flag
        self.action_shape = action_shape
        #hyper parameters
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.04
        self.batch_size = batch_size
        self.train_start = 10000
        self.dueling_option = 'avg'
        self.memory_size = memory_size
        self.memories_n = 0

        self.state_shape = (1, state_shape[0], state_shape[1])
        
        #memory
        self.replay_memory = Memory(self.memory_size)
        
        #model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())     
    
    def build_model(self):
        
        model = Sequential()
        model.add(CuDNNLSTM(300, input_shape=(self.state_shape[1], self.state_shape[2]), return_sequences=True))
        model.add(CuDNNLSTM(200))
        model.add(Dense(self.action_shape, activation='softmax'))
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model
        
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_shape)
        else:
            return np.argmax(self.model.predict(state.reshape(self.state_shape))[0])
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print('hi#####helloooooooooooooooo')
            
    def get_target_q_value(self, next_state):
        
        if self.ddqn:
            action = np.argmax(self.model.predict(next_state)[0])
            max_q_value = self.target_model.predict(next_state)[0][action]
            
        else:
            max_q_value = np.amax(self.target_model.predict(next_state)[0])
        return max_q_value
    
    def update_target_model(self, tau=0.003):
        if self.ddqn:
            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            
            for i in range(len(target_weights)):
                target_weights[i] = weights[i]*tau + target_weights[i]*(1-tau)
            self.target_model.set_weights(target_weights)
        else:
            self.target_model.set_weights(self.model.get_weights())
    
    def experience_replay(self):
        if self.replay_memory.memories_n < self.replay_memory.capacity:
            return
        
        tree_idx, minibatch = self.replay_memory.sample(self.batch_size)
        current_state = np.zeros((self.batch_size, self.state_shape[1], self.state_shape[2]))
        next_state = np.zeros((self.batch_size, self.state_shape[1], self.state_shape[2]))
        actions = np.zeros(self.batch_size, dtype=int)
        rewards = np.zeros(self.batch_size)
        done = np.zeros(self.batch_size, dtype=bool)
        qValues = np.zeros((self.batch_size, self.action_shape))
        for i in range(self.batch_size):
            current_state[i] = minibatch[i][0].reshape((self.state_shape[1], self.state_shape[2]))
            next_state[i] = minibatch[i][3].reshape((self.state_shape[1], self.state_shape[2]))
            actions[i] = minibatch[i][1]
            rewards[i] = minibatch[i][2]
            done[i] = minibatch[i][4]

            qValues[i] = self.model.predict(current_state[i].reshape(self.state_shape))
            qValue_ns = self.get_target_q_value(next_state[i].reshape(self.state_shape))

            if done[i]:
                qValues[i][actions[i]] = rewards[i]
            else:
                qValues[i][actions[i]] = rewards[i] + self.discount_factor*qValue_ns
        
        #calculating TD
        pred_qValues = np.array(self.model.predict(current_state))
        indices = np.arange(self.batch_size)
        absolute_errors = np.abs(qValues[indices, actions]-pred_qValues[indices, actions])
        self.replay_memory.batch_update(tree_idx, absolute_errors)

        #train
        self.model.fit(current_state, qValues, batch_size=self.batch_size, epochs=1, verbose=0)
        self.update_epsilon()
        
                    
        


        