import os, sys
import gym
import numpy as np
from copy import deepcopy
import tensorflow as tf
from rl4rs.utils.datautil import FeatureUtil
from rl4rs.env.slate import SlateRecEnv, SlateState
from rl4rs.env.seqslate import SeqSlateRecEnv, SeqSlateState

config = {"epoch": 10000, "maxlen": 64, "batch_size": 32, "action_size": 284,
          "class_num": 2, "dense_feature_num": 432, "category_feature_num": 21,
          "category_hash_size": 100000, "seq_num": 2, "emb_size": 128, "is_eval": False,
          "hidden_units": 128, "max_steps": 9, "action_emb_size": 32,
          "sample_file": '../rl4rs_benchmark_materials/simulator/rl4rs_dataset_a_shuf.csv',
          "model_file": "../rl4rs_benchmark_materials/simulator/finetuned/simulator_a_dien/model",
          "iteminfo_file": '../rl4rs_benchmark_materials/raw_data/item_info.csv',
          "support_rllib_mask": True, 'env': "SlateRecEnv-v0"}

config_conti = deepcopy(config)
config_conti['support_conti_env'] = True
config_conti['support_rllib_mask'] = False

sim = SlateRecEnv(config_conti, state_cls=SlateState)
env = gym.make('SlateRecEnv-v0', recsim=sim)
obs = env.reset()

batch_size = config_conti["batch_size"]

action_vec = np.full((batch_size, 32), 1)
print('size of action embedding ', np.array(env.samples.action_emb).shape)
for i in range(config["max_steps"]):
    next_obs, reward, done, info = env.step(action_vec)
    action = SlateState.get_nearest_neighbor(action_vec, env.samples.action_emb)
    print('step: ', i, ' action', action[0], ' reward: ', reward[0], ' done: ', done[0])
