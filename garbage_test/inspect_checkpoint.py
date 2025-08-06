import tensorflow as tf
import os

# --- 请修改成你的 checkpoint 文件所在的目录 ---
# 注意：路径是到目录，而不是到具体的文件
# 在你的例子中，就是 'simulator_a_dien' 目录
CHECKPOINT_DIR = '/mnt/zdy/RL4RS/rl4rs_benchmark_materials/simulator/finetuned/simulator_a_dien'

# 找到最新的 checkpoint 文件
# tf.train.latest_checkpoint 会自动处理 .index 和 .meta 文件
latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)

if not latest_ckpt:
    print(f"Error: No checkpoint found in directory: {CHECKPOINT_DIR}")
else:
    print(f"Inspecting checkpoint: {latest_ckpt}\n")

    # 使用 TensorFlow 的工具来读取 checkpoint 中的变量
    # tf.train.list_variables() 是专门为此设计的
    try:
        variable_names = tf.train.list_variables(latest_ckpt)

        print("--- Variables found in the checkpoint ---")
        for name, shape in variable_names:
            print(f"Variable Name: {name:<60} Shape: {shape}")

    except Exception as e:
        print(f"An error occurred while inspecting the checkpoint: {e}")

