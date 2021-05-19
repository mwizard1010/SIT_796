import gym
import time
from DQN import DQN
from DQN_Exploration import DQN_Exploration
from Util import save
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # fix environment of MAC OS


max_episodes = 200

def execute(env_name, agnt = DQN_Exploration, filename = 'Data/reward.csv'):
    
    env = gym.make(env_name)
    env.seed(1)

    state_shape = (1,env.observation_space.shape[0])
    agent = agnt(env=env)  # Initialise
    start_time = time.time()
    rewards = []
    episodes = []

    # for episode = 1,E do
    for episode in range(max_episodes):
        cur_state = env.reset().reshape(state_shape)
        steps = 0
        total_reward = 0
        done = False

        # for t = 1,T do
        while not done or steps < env.initial_random_steps:
            steps += 1
            action = agent.act(cur_state)  # Get action
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(state_shape)
            agent.update_model(cur_state, action, reward, new_state, done)
            cur_state = new_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        episodes.append(steps)
        print('episode {} steps: {}, reward: {},  elapsed time: {}s'.format(episode, steps, total_reward, int(time.time()-start_time)))

    save(rewards, filename)

if __name__ == "__main__":
    env_name = "LunarLander-v2"
    execute(env_name, DQN_Exploration, 'Data/Luna_DQN_Exploration.csv')
    execute(env_name, DQN, 'Data/Luna_DQN.csv')

    env_name = "MountainCar-v0"
    execute(env_name, DQN_Exploration, 'Data/Moun_DQN_Exploration.csv')
    execute(env_name, DQN, 'Data/Moun_DQN.csv')
