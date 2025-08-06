import os
import numpy as np
import gym
import ray
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rl4rs.utils.rllib_print import pretty_print
from rl4rs.nets.rllib.rllib_rawstate_model import getTFModelWithRawState
from rl4rs.nets.rllib.rllib_mask_model import getMaskActionsModel, \
    getMaskActionsModelWithRawState
from rl4rs.utils.rllib_vector_env import MyVectorEnvWrapper
from script.modelfree_trainer import get_rl_model
from rl4rs.policy.behavior_model import behavior_model
from script.offline_evaluation import ope_eval
from rl4rs.utils.fileutil import find_newest_files
import http.client
import sys
http.client.HTTPConnection._http_vsn = 10
http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

algo = 'PPO'

ray.init()

config = {"epoch": 2, "maxlen": 64, "batch_size": 64, "action_size": 284,
          "class_num": 2, "dense_feature_num": 432, "category_feature_num": 21,
          "category_hash_size": 100000, "seq_num": 2, "emb_size": 128, "is_eval": False,
          "hidden_units": 128, "max_steps": 9, "action_emb_size": 32,
          "sample_file": '../../rl4rs_benchmark_materials/simulator/rl4rs_dataset_a_shuf.csv',
          "model_file": "../../rl4rs_benchmark_materials/simulator/finetuned/simulator_a_dien/model",
          "iteminfo_file": '../../rl4rs_benchmark_materials/raw_data/item_info.csv',
          'remote_base': 'http://127.0.0.1:5000', 'trial_name': 'all',
          "support_rllib_mask": True, 'env': "SlateRecEnv-v0"}

print(config)

mask_model = getMaskActionsModel(true_obs_shape=(256,), action_size=config['action_size'])
ModelCatalog.register_custom_model("mask_model", mask_model)
mask_model_rawstate = getMaskActionsModelWithRawState(config=config, action_size=config['action_size'])
ModelCatalog.register_custom_model("mask_model_rawstate", mask_model_rawstate)
model_rawstate = getTFModelWithRawState(config=config)
ModelCatalog.register_custom_model("model_rawstate", model_rawstate)
register_env('rllibEnv-v0', lambda _: MyVectorEnvWrapper(gym.make('HttpEnv-v0', env_id=config['env'], config=config), config['batch_size']))

cfg = {
    "num_workers": 2,
    "use_critic": True,
    "use_gae": True,
    "lambda": 1.0,
    "kl_coeff": 0.2,
    "sgd_minibatch_size": 256,
    "shuffle_sequences": True,
    "num_sgd_iter": 1,
    "lr": 0.0001,
    "vf_loss_coeff": 0.5,
    "clip_param": 0.3,
    "vf_clip_param": 500.0,
    "kl_target": 0.01,
}

rllib_config = dict(
    {
        "env": "rllibEnv-v0",
        "gamma": 1,
        "explore": True,
        "exploration_config": {
            "type": "SoftQ",
            # "temperature": 1.0,
        },
        "num_gpus": 1 if config.get('gpu', True) else 0,
        "num_workers": 0,
        "framework": 'tf',
        "rollout_fragment_length": config['max_steps'],
        "batch_mode": "complete_episodes",
        "train_batch_size": min(config["batch_size"] * config['max_steps'], 1024),
        "evaluation_interval": 1,
        "evaluation_num_episodes": 2048 * 4,
        "evaluation_config": {
            "explore": False
        },
        "log_level": "INFO",
    },
    **cfg)
print('rllib_config', rllib_config)
trainer = get_rl_model(algo, rllib_config)

# restore_file = ''
# trainer.restore(restore_file)

for i in range(config["epoch"]):
    result = trainer.train()
    if (i + 1) % 1 == 0 or i == 0:
        print(pretty_print(result))


ray.shutdown()