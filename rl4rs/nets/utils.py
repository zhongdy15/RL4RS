import numpy as np
import tensorflow as tf
from deepctr.layers.sequence import AttentionSequencePoolingLayer, DynamicGRU
from tensorflow.keras import layers, regularizers


def id_input_processing(category_feature_input, config):
    emb_size = config['emb_size']
    category_hash_size = config['category_hash_size']
    emb_layer = layers.Embedding(input_dim=category_hash_size, output_dim=emb_size)
    category_emb = emb_layer(category_feature_input)
    category_feature = layers.GlobalAveragePooling1D()(category_emb)
    return category_feature


def id_input_processing_attn(category_feature_input, config):
    emb_size = config['emb_size']
    hidden_unit = config['hidden_units']
    category_hash_size = config['category_hash_size']
    emb_layer = layers.Embedding(input_dim=category_hash_size, output_dim=emb_size)
    category_emb = emb_layer(category_feature_input)
    category_feature = tf.keras.layers.Attention()([category_emb, category_emb])
    category_feature = tf.keras.layers.GlobalAveragePooling1D()(category_feature)
    category_feature_2 = layers.Flatten()(category_emb)
    return layers.Concatenate(axis=-1)([category_feature, category_feature_2])


def id_input_processing_lstm(category_feature_input, config):
    emb_size = config['emb_size']
    hidden_unit = config['hidden_units']
    category_hash_size = config['category_hash_size']
    emb_layer = layers.Embedding(input_dim=category_hash_size, output_dim=emb_size)
    category_emb = emb_layer(category_feature_input)
    category_feature = layers.GRU(units=hidden_unit)(category_emb)
    category_feature_2 = layers.Flatten()(category_emb)
    return layers.Concatenate(axis=-1)([category_feature, category_feature_2])


def id_input_processing_concat(category_feature_input, config):
    emb_size = config['emb_size']
    category_hash_size = config['category_hash_size']
    emb_layer = layers.Embedding(input_dim=category_hash_size, output_dim=emb_size)
    category_emb = emb_layer(category_feature_input)
    category_feature = layers.Flatten()(category_emb)
    return category_feature


def dense_input_processing(cross_feature_input, config):
    hidden_unit = config['hidden_units']
    cross_feature = layers.Dense(hidden_unit, activation=layers.ELU())(cross_feature_input)
    cross_feature = layers.Dropout(0.2)(cross_feature)
    cross_feature = layers.Dense(hidden_unit, activation=layers.ELU())(cross_feature)
    cross_feature = layers.Dropout(0.2)(cross_feature)
    return cross_feature


def sequence_input_concat(sequence_feature_input, config):
    category_hash_size = config['category_hash_size']
    hidden_unit = config['hidden_units']
    emb_size = config['emb_size']
    seq_num = config['seq_num']

    seq_index_layer = layers.Lambda(lambda x: x[0][:, x[1]])
    emb_layer = layers.Embedding(input_dim=category_hash_size, output_dim=emb_size)

    seqs_lstm = []
    for i in range(seq_num):
        seq_i = seq_index_layer([sequence_feature_input, i])
        seq_i_embeddings = emb_layer(seq_i)
        seq_i_lstm = layers.GlobalAveragePooling1D()(seq_i_embeddings)
        seqs_lstm.append(seq_i_lstm)

    seqs_embeddings = layers.Concatenate(axis=-1)(seqs_lstm) if len(seqs_lstm) > 1 else seqs_lstm[0]

    return seqs_embeddings


def sequence_input_LSTM(sequence_feature_input, config):
    category_hash_size = config['category_hash_size']
    hidden_unit = config['hidden_units']
    emb_size = config['emb_size']
    seq_num = config['seq_num']

    seq_index_layer = layers.Lambda(lambda x: x[0][:, x[1]])

    emb_layer = layers.Embedding(input_dim=category_hash_size, output_dim=emb_size)

    seqs_lstm = []
    for i in range(seq_num):
        seq_i = seq_index_layer([sequence_feature_input, i])
        seq_i_embeddings = emb_layer(seq_i)
        seq_i_lstm = layers.GRU(units=hidden_unit)(seq_i_embeddings)
        seqs_lstm.append(seq_i_lstm)

    seqs_embeddings = layers.Concatenate(axis=-1)(seqs_lstm) if len(seqs_lstm) > 1 else seqs_lstm[0]

    return seqs_embeddings


def sequence_input_attn(input, config):
    category_hash_size = config['category_hash_size']
    hidden_unit = config['hidden_units']
    emb_size = config['emb_size']
    maxlen = config['maxlen']
    batch_size = config['batch_size']
    seq_num = config['seq_num']

    sequence_feature_input = input[0]
    id_slate_input = input[1]

    sequence_length = tf.fill((tf.shape(sequence_feature_input)[0], 1), maxlen)
    seq_index_layer = layers.Lambda(lambda x: x[0][:, x[1]])
    emb_layer = layers.Embedding(input_dim=category_hash_size, output_dim=emb_size)
    id_slate_embeddings = emb_layer(id_slate_input)
    id_slate_pooling = tf.math.reduce_mean(id_slate_embeddings, axis=1, keepdims=True)
    seqs_attn = []
    for i in range(seq_num):
        seq_i = seq_index_layer([sequence_feature_input, i])
        seq_i_embeddings = emb_layer(seq_i)

        # --- 关键部分：根据你发现的命名规律动态生成层名 ---
        if i == 0:
            # i=0 时的特殊命名
            rnn_layer_name = "dynamic_gru"
            attn_layer_name = "attention_sequence_pooling_layer"
            permute_layer_name = "permute"  # Permute 层也可能有类似规则，先假设是这样
            augru_layer_name = "dynamic_gru_1"
        else:
            # i > 0 时的通用命名规则
            # 注意这里的数字可能需要根据你的发现微调
            # 第一个 GRU: i=1 -> _2, i=2 -> _4, i=3 -> _6 ... (似乎是 2*i)
            rnn_layer_name = f"dynamic_gru_{2 * i}"

            # Attention 层: i=1 -> _1, i=2 -> _2, ... (似乎是 i)
            # 注意，Keras 可能会把 attention_sequence_pooling_layer 变成 attention_sequence_pooling_layer_1
            # 所以这里的命名可能是 attention_sequence_pooling_layer_{i}
            attn_layer_name = f"attention_sequence_pooling_layer_{i}"

            # Permute 层: i=1 -> _1, ... (似乎是 i)
            permute_layer_name = f"permute_{i}"

            # 第二个 GRU: i=1 -> _3, i=2 -> _5, ... (似乎是 2*i + 1)
            augru_layer_name = f"dynamic_gru_{2 * i + 1}"

        rnn_outputs = DynamicGRU(emb_size, return_sequence=True, name=rnn_layer_name)([seq_i_embeddings, sequence_length])
        scores = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), return_score=True, name=attn_layer_name)([
            id_slate_pooling, rnn_outputs, sequence_length])
        final_state2 = DynamicGRU(emb_size * 2, gru_type='AUGRU', return_sequence=False, name=augru_layer_name
                                  )([rnn_outputs, sequence_length, tf.keras.layers.Permute([2, 1])(scores)])
        seqs_attn.append(final_state2)

    seqs_embeddings = layers.Concatenate(axis=-1)(seqs_attn) if len(seqs_attn) > 1 else seqs_attn[0]

    return tf.squeeze(seqs_embeddings, axis=1)
