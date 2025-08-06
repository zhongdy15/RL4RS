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

from copy import deepcopy
config_rawstate = deepcopy(config)
config_rawstate['support_rllib_mask'] = True
config_rawstate['support_conti_env'] = True
config_rawstate['rawstate_as_obs'] = False
sim = SlateRecEnv(config_rawstate, state_cls=SlateState)
env = gym.make('SlateRecEnv-v0', recsim=sim)
obs = env.reset()
print('observation type: ', type(obs[0]))
print('size of obs.action_mask: ', len(obs[0]['action_mask']))
# print('size of obs.category_feature: ', len(obs[0]['category_feature']))
# print('size of obs.dense_feature: ', len(obs[0]['dense_feature']))
# print('size of obs.sequence_feature: ', obs[0]['sequence_feature'].shape)

batch_size = config["batch_size"]

action_vec = np.full((batch_size, 32), 1)
print('size of action embedding ', np.array(env.samples.action_emb).shape)
for i in range(config["max_steps"]):
    next_obs, reward, done, info = env.step(action_vec)
    action = SlateState.get_nearest_neighbor(action_vec, env.samples.action_emb)
    print('step: ', i, ' action', action[0], ' reward: ', reward[0], ' done: ', done[0])
