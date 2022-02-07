# -*- coding: utf-8 -*
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
import gym
import numpy as np
import pickle
from Config import DDQNConfig, DQfDConfig, MineRL_DQfDConfig as Config,MineRL_DQfDConfig
from DQfD_V3 import DQfD
from DQfDDDQN import DQfDDDQN
from collections import deque
import itertools
from Models import make_vae
import torch
import torch.nn.functional as F

def map_scores(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


def run_DDQN(index, env):
    with tf.compat.v1.variable_scope('DDQN_' + str(index)):
        agent = DQfDDDQN(env, DDQNConfig())
    scores = []
    for e in range(Config.episode):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive([state, action, reward, next_state, done, 0.0])  # 0. means it is not a demo data
            agent.train_Q_network(update=False)
            state = next_state
        if done:
            scores.append(score)
            agent.sess.run(agent.update_target_net)
            print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
                  "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
            # if np.mean(scores[-min(10, len(scores)):]) > 490:
            #     break
    return scores



def declassify_cam(x):
    import pandas as pd 

    if isinstance(x, pd.core.series.Series):
        x = x.values
    s = 10
    # quadrants
    if x == 0:
        return [0,0]
    elif x == 2:
        return [s,0]
    elif x == 4:
        return [0,-s]
    elif x == 6:
        return [-s,0]
    else:
        return [0,s]

def get_obs_enc(obs):

#         obs = obs.transpose(-1,0,1).copy()
    img = smooth_obs(obs)
    _ , mu , _ = vae_model(img.unsqueeze(0).to(device))
    mu = mu.detach().cpu().numpy()[0]
    return mu

def proc_action(act,lbe):

    act = np.array(lbe.inverse_transform([act])[0].split("_")).astype(int)

    action = {}

    action['camera'] = declassify_cam(act[3])
    action['attack'] = act[2]

    action['forward'] = act[0]
    action['jump'] = act[1]

    return action

def run_DQfD(index, env):
    vae_model, obs2state = make_vae()
    with open("minerl_lbe.pkl", 'rb') as f:
        lbe = pickle.load(f)
    
    with open(Config.DEMO_DATA_PATH, 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.demo_buffer_size))
        assert len(demo_transitions) == Config.demo_buffer_size
    with tf.compat.v1.variable_scope('DQfD_' + str(index)):
        agent = DQfD(env, MineRL_DQfDConfig(), demo_transitions=demo_transitions)

    agent.pre_train()  # use the demo data to pre-train network
    scores, e, replay_full_episode = [], 0, None
    while True:
        done, score, n_step_reward, state = False, 0, None, obs2state(env.reset()['pov'])

        t_q = deque(maxlen=Config.trajectory_n)
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ =  env.step(proc_action(action,lbe = lbe))
            next_state = obs2state(next_state['pov'])
            score += reward
            reward = reward if not done or score == 499 else -100
            reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][2]  # record the earliest reward for the sub
            t_q.append([state, action, reward, next_state, done, 0.0])
            if len(t_q) == t_q.maxlen:
                if n_step_reward is None:  # only compute once when t_q first filled
                    n_step_reward = sum([t[2]*Config.GAMMA**i for i, t in enumerate(t_q)])
                else:
                    n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
                    n_step_reward += reward*Config.GAMMA**(Config.trajectory_n-1)
                t_q[0].extend([n_step_reward, next_state, done, t_q.maxlen])  # actual_n is max_len here
                agent.perceive(t_q[0])  # perceive when a transition is completed
                if agent.replay_memory.full():
                    agent.train_Q_network(update=False)  # train along with generation
                    replay_full_episode = replay_full_episode or e
            state = next_state
        if done:
            # handle transitions left in t_q
            t_q.popleft()  # first transition's n-step is already set
            transitions = set_n_step(t_q, Config.trajectory_n)
            for t in transitions:
                agent.perceive(t)
                if agent.replay_memory.full():
                    agent.train_Q_network(update=False)
                    replay_full_episode = replay_full_episode or e
            if agent.replay_memory.full():
                scores.append(score)
                agent.sess.run(agent.update_target_net)
            if replay_full_episode is not None:
                print("episode: {}  trained-episode: {}  score: {}  memory length: {}  epsilon: {}"
                      .format(e, e-replay_full_episode, score, len(agent.replay_memory), agent.epsilon))
            # if np.mean(scores[-min(10, len(scores)):]) > 495:
            #     break
            # agent.save_model()
        if len(scores) >= Config.episode:
            break
        e += 1
    return scores


# extend [n_step_reward, n_step_away_state] for transitions in demo
def set_n_step(container, n):
    t_list = list(container)
    # accumulated reward of first (trajectory_n-1) transitions
    n_step_reward = sum([t[2] * Config.GAMMA**i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
    for begin in range(len(t_list)):
        end = min(len(t_list) - 1, begin + Config.trajectory_n - 1)
        n_step_reward += t_list[end][2]*Config.GAMMA**(end-begin)
        # extend[n_reward, n_next_s, n_done, actual_n]
        t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end-begin+1])
        n_step_reward = (n_step_reward - t_list[begin][2])/Config.GAMMA
    return t_list


def get_demo_data():
    import pandas as pd 
    import numpy as np
    from tqdm import tqdm
    import pickle 
    
    with open("minerl_data.pkl", 'rb') as f:
        embdat = pickle.load(f)
    
    demo = deque()
    episode = []
    for i in tqdm(range(embdat.shape[0]-1)):
        row = embdat.iloc[i]
        done = row.done
        
        episode.append([row.enc,row.action,row.reward,
            embdat.iloc[i+1].enc,done,1.0])

        if done:
            episode = set_n_step(episode, Config.trajectory_n)
            demo.extend(episode)
            episode = []


    with open(Config.DEMO_DATA_PATH, 'wb') as f:
        pickle.dump(demo, f, protocol=2)


if __name__ == '__main__':
    import minerl

    env = gym.make(Config.ENV_NAME)
    

    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    
    # get_demo_data()
    # ----------------------------- get DQfD scores --------------------------------
    
    dqfd_sum_scores = np.zeros(Config.episode)
    for i in range(Config.iteration):
        scores = run_DQfD(i, env)
        dqfd_sum_scores = np.array([a + b for a, b in zip(scores, dqfd_sum_scores)])
    dqfd_mean_scores = dqfd_sum_scores / Config.iteration
    with open('./dqfd_mean_scores.p', 'wb') as f:
        pickle.dump(dqfd_mean_scores, f, protocol=2)

    map_scores(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores,
        xlabel='Red: dqfd         Blue: ddqn', ylabel='Scores')
    env.close()
    # gym.upload('/tmp/carpole_DDQN-1', api_key='sk_VcAt0Hh4RBiG2yRePmeaLA')


