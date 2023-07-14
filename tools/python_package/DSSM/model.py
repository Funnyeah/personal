# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/6/25 15:18
@summary:
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input
# from match.layers.modules import DNN

from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Conv1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Concatenate, Activation, GlobalAveragePooling2D, Reshape
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.initializers import Zeros


class DNN(Layer):
    """DNN Layer"""
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., **kwargs):
        """
        DNN part
        :param hidden_units: A list. List of hidden layer units's numbers 各个隐藏层的神经元数量，即各层输出大小
        :param activation: A string. Activation function 激活函数
        :param dnn_dropout: A scalar. dropout number dropout比率
        """
        super(DNN, self).__init__(**kwargs)
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x
    
class Dssm(Model):
    def get_config(self):
        return {"user_sparse_feature_columns": self.user_sparse_feature_columns,
                "item_sparse_feature_columns": self.item_sparse_feature_columns,
                "dnn_dropout": self.dnn_dropout}

    def __init__(self, user_sparse_feature_columns, item_sparse_feature_columns, user_dense_feature_columns=(),
                 item_dense_feature_columns=(), num_sampled=1,
                 user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32), dnn_activation='relu',
               l2_reg_embedding=1e-6, dnn_dropout=0, **kwargs):
        super(Dssm, self).__init__(**kwargs)
        self.num_sampled = num_sampled
        self.user_sparse_feature_columns = user_sparse_feature_columns  #用户稀疏特征列
        self.user_dense_feature_columns = user_dense_feature_columns   #用户稠密特征列
        self.item_sparse_feature_columns = item_sparse_feature_columns #物品稀疏特征列
        self.item_dense_feature_columns = item_dense_feature_columns  #物品稠密特征列
        self.dnn_dropout = dnn_dropout

         # 用户和物品稀疏特征的嵌入层
        self.user_embed_layers = {
            'embed_' + str(feat['feat']): Embedding(input_dim=feat['feat_num'],
                                         input_length=feat['feat_len'],
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(l2_reg_embedding))
            for feat in self.user_sparse_feature_columns
        }

        self.item_embed_layers = {
            'embed_' + str(feat['feat']): Embedding(input_dim=feat['feat_num'],
                                         input_length=feat['feat_len'],
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(l2_reg_embedding))
            for feat in self.item_sparse_feature_columns
        }

         # 用户和物品DNN部分
        self.user_dnn = DNN(user_dnn_hidden_units, dnn_activation, dnn_dropout)
        self.item_dnn = DNN(item_dnn_hidden_units, dnn_activation, dnn_dropout)

    def cosine_similarity(self, tensor1, tensor2):
        """计算cosine similarity"""
        # 把张量拉成矢量，这是我自己的应用需求
        tensor1 = tf.reshape(tensor1, shape=(1, -1))
        tensor2 = tf.reshape(tensor2, shape=(1, -1))
        # 求模长
        tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1)))
        tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2)))
        # 内积
        tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2))
        # cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
        cosin = tf.realdiv(tensor1_tensor2, tensor1_norm * tensor2_norm)

        return cosin

    def call(self, inputs, training=None, mask=None):
        """
        TODO：
        加入连续特征直接concat拼接输入DNN
        user_continue_inputs, item_continue_inputs = inputs
        
        user_continue_feat = tf.concat(user_continue_inputs, axis=-1)
        user_feat = tf.concat([user_continue_inputs,user_continue_feat], axis=-1)
        """
        user_sparse_inputs, item_sparse_inputs = inputs
        # user tower
        # 拼接所有稀疏特征
        user_sparse_embed = tf.concat([self.user_embed_layers['embed_{}'.format(k)](v)
                                  for k, v in user_sparse_inputs.items()], axis=-1)

        # 喂入DNN
        user_dnn_input = user_sparse_embed
        self.user_dnn_out = self.user_dnn(user_dnn_input)
        
        # item tower
        item_sparse_embed = tf.concat([self.item_embed_layers['embed_{}'.format(k)](v)
                                       for k, v in item_sparse_inputs.items()], axis=-1)
        item_dnn_input = item_sparse_embed
        self.item_dnn_out = self.item_dnn(item_dnn_input)
        
        # 计算用户和物品的余弦相似度
        cosine_score = self.cosine_similarity(self.item_dnn_out, self.user_dnn_out)
        
        # 经sigmoid后输出
        output = tf.reshape(tf.sigmoid(cosine_score), (-1, 1))

        return output

    def build_graph(self, **kwargs):
        """
        入口
        """
        
        # 构建用户稀疏特征的输入层
        user_sparse_inputs = {uf['feat']: Input(shape=(1, ), dtype=tf.float32) for uf in
                              self.user_sparse_feature_columns}
        # 构建物品稀疏特征的输入层
        item_sparse_inputs = {uf['feat']: Input(shape=(1, ), dtype=tf.float32) for uf in
                              self.item_sparse_feature_columns}
        
        # 定义模型输入输出
        model = Model(inputs=[user_sparse_inputs, item_sparse_inputs],
              outputs=self.call([user_sparse_inputs, item_sparse_inputs]))
        
        # 参数设置
        model.__setattr__("user_input", user_sparse_inputs)
        model.__setattr__("item_input", item_sparse_inputs)
        model.__setattr__("user_embed", self.user_dnn_out)
        model.__setattr__("item_embed", self.item_dnn_out)
        return model

# def model_test():
#     user_features = [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
#     item_features = [{'feat': 'item_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
#     model = Dssm(user_features, item_features)
#     model.build_graph()
#
# model_test()