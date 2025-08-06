import gym
import traceback # 用于打印异常信息，但通常情况下，如果只在except块中使用，可以不放在顶部。为明确起见，此处保留。
import rl4rs

# 提供的配置
config = {
    "epoch": 2, "maxlen": 64, "batch_size": 64, "action_size": 284,
    "class_num": 2, "dense_feature_num": 432, "category_feature_num": 21,
    "category_hash_size": 100000, "seq_num": 2, "emb_size": 128, "is_eval": False,
    "hidden_units": 128, "max_steps": 9, "action_emb_size": 32,
    "sample_file": '../../rl4rs_benchmark_materials/simulator/rl4rs_dataset_a_shuf.csv',
    "model_file": "../../rl4rs_benchmark_materials/simulator/finetuned/simulator_a_dien/model",
    "iteminfo_file": '../../rl4rs_benchmark_materials/raw_data/item_info.csv',
    'remote_base': 'http://127.0.0.1:5000', # 这是关键，你的HTTP服务应该在这个地址运行
    'trial_name': 'all',
    "support_rllib_mask": True,
    'env': "SlateRecEnv-v0" # 这个是传递给远程环境的env_id
}

print("--- 开始测试 HttpEnv ---")
env_id = 'HttpEnv-v0'  # 使用我们注册的ID
env = None  # 初始化 env 为 None，确保finally块中可以安全检查

try:
    # 1. 实例化环境
    print(f"\n尝试使用 gym.make('{env_id}', config={config['env']}) 实例化环境...")
    # 注意：gym.make的第二个参数是 kwargs，所以config需要通过键值对传入
    env = gym.make(env_id, env_id=config['env'], config=config)
    print(f"环境 '{env_id}' 实例化成功！")
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")

    # 2. 测试 reset 功能
    print("\n--- 测试 reset 功能 ---")
    observation = env.reset()
    print(f"首次 reset 后的观测: {observation}")
    print(f"观测类型: {type(observation)}")
    # assert env.observation_space.contains(observation), "Reset后的观测不在观测空间内！"

    # 3. 测试 step 功能
    print("\n--- 测试 step 功能 ---")
    num_episodes = 2  # 测试运行2个episode
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        observation = env.reset()  # 每个episode开始时重置环境
        done = False
        total_reward = 0
        step_count = 0

        # 运行一个episode直到done为True或者达到最大步数
        max_steps_per_episode = config.get("max_steps", 100)  # 从config获取或默认100步

        while not done and step_count < max_steps_per_episode:
            # 从动作空间中随机采样一个动作
            action = [0 for _ in range(64)]

            print(f"Step {step_count + 1}: 采取动作 {action} ")

            # 执行一步
            next_observation, reward, done, info = env.step(action)

            # total_reward = total_reward+reward
            step_count += 1

            # print(
            #     f"  下一个观测: (shape: {next_observation.shape if isinstance(next_observation, np.ndarray) else 'N/A'}, type: {type(next_observation)})")
            print(f"  奖励: {reward}")
            print(f"  是否结束 (done): {done}")
            print(f"  信息 (info): {info}")

            # 验证下一个观测是否在观测空间内
            # assert env.observation_space.contains(next_observation), f"Step后的观测不在观测空间内！ (Step {step_count})"

            observation = next_observation

        print(f"Episode {episode + 1} 结束. 总步数: {step_count}, 总奖励: {total_reward:.4f}, Done状态: {done}")

except Exception as e:
    print(f"\n测试过程中发生错误: {e}")
    traceback.print_exc() # 使用 traceback 打印详细异常信息

finally:
    # 4. 测试 close 功能
    print("\n--- 测试 close 功能 ---")
    if env is not None:
        env.close()
        print("环境已关闭。")
    else:
        print("环境未成功创建，无需关闭。")

print("\n--- HttpEnv 基本功能测试完成 ---")
