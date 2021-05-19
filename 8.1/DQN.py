#reference: https://keon.io/deep-q-learning/

import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQN:
    def __init__(self, env):
        self.env = env
        

        self.gamma = 0.99 # discount rate 
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.05
        self.update_target_counter = 0
        self.C = 8 # interval for updating target network
        self.initial_random_steps = 2000
        self.actions_count = 0

        # self.clip_errors = True

        # memory M
        self.memory = deque(maxlen=2000)
        # Initialize Q-network Q
        self.model = self._build_model()
        # Initialize target Q-network Qˆ 
        self.target_model = self._build_model()

    def observation_space(self):
        return self.env.observation_space

    def action_space(self):
        return self.env.action_space

    def _build_model(self):
        model = Sequential()
        state_shape = self.observation_space().shape
        model.add(Dense(24, input_shape=state_shape, activation="relu"))
        model.add(Dense(self.action_space().n, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        self.actions_count += 1
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.action_space().sample()
        return np.argmax(self.model.predict(state)[0])

    def update_model(self, state, action, reward, new_state, done):
        # Store transition (st, at, rt, st+1) in M
        self.memorize(state, action, reward, new_state, done)
        self.fit_model()
        self.update_target_model()

    def fit_model(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return # do nothing

        samples = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, new_state, done in samples:
            target = reward
            target_f = self.target_model.predict(state)
            predict_f = self.model.predict(state)
            if not done:
                #update Bellman equation
                target = reward + self.gamma * np.amax(self.target_model.predict(new_state)[0])

                # if self.clip_errors:
                    #clip error to -1, +1
                    # if (target[0][action] > predict[0][action]):
                    #     target[0][action] = predict[0][action] + 1
                    # elif (target[0][action] < predict[0][action]):
                    #     target[0][action] = predict[0][action] - 1
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)

        states = np.concatenate(states,axis=0)
        targets = np.concatenate(targets,axis=0)
        self.model.fit(states, targets, epochs=1, verbose=0)

    def update_target_model(self):
        self.update_target_counter += 1
        if (self.update_target_counter > self.C):
            self.update_target_counter = 0
            self.target_model.set_weights(self.model.get_weights())
