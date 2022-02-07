# -*- coding: utf-8 -*
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
import gym
import numpy as np
import pickle
from Config import Config, DDQNConfig, DQfDConfig
from DQfD_V3 import DQfD
from DQfDDDQN import DQfDDDQN
from collections import deque
import itertools


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


def run_DQfD(index, env):
    with open(Config.DEMO_DATA_PATH, 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.demo_buffer_size))
        assert len(demo_transitions) == Config.demo_buffer_size
    with tf.compat.v1.variable_scope('DQfD_' + str(index)):
        agent = DQfD(env, DQfDConfig(), demo_transitions=demo_transitions)

    agent.pre_train()  # use the demo data to pre-train network
    scores, e, replay_full_episode = [], 0, None
    while True:
        done, score, n_step_reward, state = False, 0, None, env.reset()
        t_q = deque(maxlen=Config.trajectory_n)
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
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


def get_demo_data(env):
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # agent.restore_model()
    with tf.compat.v1.variable_scope('get_demo_data'):
        agent = DQfDDDQN(env, DDQNConfig())

    e = 0
    while True:
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        demo = []
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive([state, action, reward, next_state, done, 0.0])  # 0. means it is not a demo data
            demo.append([state, action, reward, next_state, done, 1.0])  # record the data that could be expert-data
            agent.train_Q_network(update=False)
            state = next_state
        if done:
            if score == 500:  # expert demo data
                demo = set_n_step(demo, Config.trajectory_n)
                agent.demo_buffer.extend(demo)
            agent.sess.run(agent.update_target_net)
            print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
                  "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
            if len(agent.demo_buffer) >= Config.demo_buffer_size:
                agent.demo_buffer = deque(itertools.islice(agent.demo_buffer, 0, Config.demo_buffer_size))
                break
        e += 1

    with open(Config.DEMO_DATA_PATH, 'wb') as f:
        pickle.dump(agent.demo_buffer, f, protocol=2)


if __name__ == '__main__':
    env = gym.make(Config.ENV_NAME)
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # ------------------------ get demo scores by DDQN -----------------------------
    # get_demo_data(env)
    # --------------------------  get DDQN scores ----------------------------------
    # ddqn_sum_scores = np.zeros(Config.episode)
    # for i in range(Config.iteration):
    #     scores = run_DDQN(i, env)
    #     ddqn_sum_scores = np.array([a + b for a, b in zip(scores, ddqn_sum_scores)])
    # ddqn_mean_scores = ddqn_sum_scores / Config.iteration
    # with open('./ddqn_mean_scores.p', 'wb') as f:
    #     pickle.dump(ddqn_mean_scores, f, protocol=2)
    with open('./ddqn_mean_scores.p', 'rb') as f:
        ddqn_mean_scores = pickle.load(f)
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


