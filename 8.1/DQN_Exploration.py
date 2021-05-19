import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from DQN import DQN
from scipy import stats

class DQN_Exploration(DQN):
    def __init__(self, env):
        self.env = env
        super().__init__(env)
        self.memory = deque(maxlen=10000)

        # Initialize dynamics predictor D
        self.dynamics_model = self._build_dynamics_model()

        self.initial_random_steps = 4000
        self.update_count = 0
        self.converged = False

    def update_model(self, state, action, reward, new_state, done):
        super().update_model(state, action, reward, new_state, done)
        self.update_count += 1
        if self.update_count % 25 == 0:
            self.fit_dynamics_model()
        if self.update_count == 1000:
            self.eval_dynamics_model()

    def act(self, state):
        self.actions_count += 1
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        # Explore = True with probability ε
        if np.random.random() < self.epsilon:
            return self.explore(state)
        # Explore = False 
        return np.argmax(self.model.predict(state)[0])

    def explore(self,state):
        if not self.converged:
            return self.action_space().sample()

        N = len(self.memory)
        num_past_samples = 25
        samples = []
        # Retrieve the last 25 F states visited from transitions in M
        for i in range(N- num_past_samples ,N):
           samples.append(self.memory[i][0])  # add state

        min_p = np.inf
        action = -1
        for a in range(self.action_space().n):
            #  D(St,a)
            next_state = self.dynamics_model.predict(np.append(state, [[a]], axis=1))
            p = self._probability(next_state, samples)
            if p < min_p: #argmin
                action = a
                min_p = p
        return action 
    def _probability(self,state, samples):
        li = [x[0] for x in samples]

        # mean 
        mean = np.mean(li,axis = 0)

        # covariance 
        li = np.stack(li).T
        cov = np.cov(li)

        p = stats.multivariate_normal.pdf(state[0], mean, cov, allow_singular=True)
        return p

    def _build_dynamics_model(self):
        model = Sequential()
        input_shape = (self.observation_space().shape[0] + 1,)
        model.add(Dense(24, input_shape=input_shape, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.observation_space().shape[0], activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.02))
        return model

    def fit_dynamics_model(self):
        batch_size = 64
        if len(self.memory) < batch_size:
            return [], []# do nothing
        states, targets = self._get_from_patch_dynamic(batch_size)
        self.dynamics_model.fit(states, targets, epochs=1, verbose=0)

    def eval_dynamics_model(self):
        batch_size = 64
        states, targets = self._get_from_patch_dynamic(64)
        scores = self.dynamics_model.evaluate(states, targets, verbose=0)
        if scores < 0.005:
            self.converged = True
            print('Dynamics model has converged!')
        print(self.dynamics_model.metrics_names, scores)

    def _get_from_patch_dynamic(self, batch_size): 
        samples = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, new_state, done in samples:
            input_state = np.append(state, [[action]], axis=1)
            target = new_state
            states.append(input_state)
            targets.append(target)

        states = np.concatenate(states,axis=0)
        targets = np.concatenate(targets,axis=0)
        return states, targets
