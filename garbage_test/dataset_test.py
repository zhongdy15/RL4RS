import os, sys
import gym
import numpy as np
from copy import deepcopy
import tensorflow as tf
from rl4rs.utils.datautil import FeatureUtil
from rl4rs.env.slate import SlateRecEnv, SlateState
from rl4rs.env.seqslate import SeqSlateRecEnv, SeqSlateState

config = {"epoch": 10000, "maxlen": 64, "batch_size": 64, "action_size": 284,
          "class_num": 2, "dense_feature_num": 432, "category_feature_num": 21,
          "category_hash_size": 100000, "seq_num": 2, "emb_size": 128, "is_eval": False,
          "hidden_units": 128, "max_steps": 9, "action_emb_size": 32,
          "sample_file": '/project/wangkai/rl4rs_benchmark_materials/simulator/rl4rs_dataset_a_shuf.csv',
          "model_file": "/project/wangkai/rl4rs_benchmark_materials/simulator/finetuned/simulator_a_dien/model",
          "iteminfo_file": '/project/wangkai/rl4rs_benchmark_materials/raw_data/item_info.csv',
          "support_rllib_mask": True, 'env': "SlateRecEnv-v0"}

sim = SlateRecEnv(config, state_cls=SlateState)
env = gym.make('SlateRecEnv-v0', recsim=sim)

obs = env.reset()
print('batchsize of batched environment: ', len(obs))

for i in range(config["max_steps"]):
    action = env.offline_action
    next_obs, reward, done, info = env.step(action)
    print('step: ', i, ' action', action[0], ' reward: ', reward[0], ' offline reward: ', env.offline_reward[0],
          ' done: ', done[0])

print('observation type: ', type(next_obs[0]))
print('size of obs.action_mask: ', len(next_obs[0]['action_mask']))
print('size of obs.obs: ', len(next_obs[0]['obs']))