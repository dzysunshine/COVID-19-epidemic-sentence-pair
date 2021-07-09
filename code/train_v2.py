import numpy as np
import pandas as pd
import os
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from bert4keras.backend import keras, K,search_layer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from sklearn.model_selection import StratifiedKFold

# 确定随机种子
np.random.seed(2020)

# 确定主目录路径
main_path = os.getcwd()
# 读取训练集数据
train_df = pd.read_csv(main_path + '/data/Dataset/train.csv')
valid_df = pd.read_csv(main_path + '/data/Dataset/dev.csv')

# train_df = pd.read_csv('../data/Dataset/train.csv')
# valid_df = pd.read_csv('../data/Dataset/dev.csv')

print(f'训练集数据共有：{len(train_df)}条')
print(f'验证集数据共有：{len(valid_df)}条')

# 训练阶段没有用到测试集的数据
# test_df = pd.read_csv('../data/Dataset/test.csv')

# 这里使用了2019年平安医疗科技疾病问答迁移学习比赛的数据
train_ext_df = pd.read_csv(main_path +'/data/External/chip2019.csv')
# train_ext_df = pd.read_csv('../data/External/chip2019.csv')

# 将训练集中有缺失值的行删去（实际上并没有缺失的）
train_df.dropna(axis=0, inplace=True)
print(f'删除后的训练集数据共有：{len(train_df)}条')

# 取出训练集和验证集中的值（将dataframe转换成了ndarray）
train_data = train_df[['query1','query2','label']].values
valid_data = valid_df[['query1','query2','label']].values

# 这里的test_data在训练阶段其实是没有用到的
# test_data = test_df[['query1','query2','label']].values
train_ext_data = train_ext_df[['question1','question2','label']].values
print(f'额外的训练集数据共有：{len(train_ext_data)}条')

train_ext_data = np.concatenate([train_data, train_ext_data], axis = 0)
print(f'原训练集数据与额外训练集数据拼接后共有：{len(train_ext_data)}条')
print('--' * 15 + '数据已加载完成'+ '--' * 15)

def build_model(mode='bert',filename='bert',lastfour=False,LR=1e-5,DR=0.2):
    
    # 定义配置文件，模型加载文件和词表的路径
    path = main_path + '/data/External/' + filename + '/'
    config_path = path + 'bert_config.json'
    checkpoint_path = path + 'bert_model.ckpt'
    dict_path = path + 'vocab.txt'
    # 定义一个全局变量 tokenizer
    global tokenizer

    # 使用预训练模型对应的分词方法
    tokenizer = Tokenizer(dict_path,do_lower_case=True)

    # 定义模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        # bert 是否包含pool部分
        with_pool=True,
        model=mode,
        return_keras_model=False,
    )
    # 对是否使用最后四层的情况分别处理
    if lastfour:
        model = Model(
            inputs=bert.model.input,
            outputs=[
                bert.model.layers[-3].get_output_at(0),
                bert.model.layers[-11].get_output_at(0),
                bert.model.layers[-19].get_output_at(0),
                bert.model.layers[-27].get_output_at(0)
            ]
        )
        # 输出维度是768
        output = model.outputs
        # 把最后四层（共12层）的【CLS】取出来
        output1 = Lambda(lambda x: x[:,0], name='Pooler1')(output[0])
        output2 = Lambda(lambda x: x[:,0], name='Pooler2')(output[1])
        output3 = Lambda(lambda x: x[:,0], name='Pooler3')(output[2])
        output4 = Lambda(lambda x: x[:,0], name='Pooler4')(output[3])
        # 将最后四层的结果拼接起来
        output = Concatenate(axis=1)([output1, output2, output3, output4])
        print(f'将后四层的CLS拼接后的维度为：{output.shape}')
    else:
        output = bert.model.output

    output = Dropout(rate=DR)(output)
    output = Dense(units=2,
                   activation='softmax',
                   kernel_initializer=bert.initializer)(output)

    model = Model(bert.model.input, output)

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = Adam(LR),
        metrics = ['accuracy']
    )
    return model
# 定义数据生成器
class data_generator(object):
    def __init__(self,data,batch_size=32,random=True):
        self.data = data
        self.batch_size = batch_size
        self.random = random
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self,random = False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids,batch_segment_ids,batch_labels = [],[],[]
        for i in idxs:
            text1,text2,label = self.data[i]
            token_ids,segment_ids = tokenizer.encode(text1, text2,maxlen=64)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids,batch_labels = [], [], []

    def forfit(self):
        while True:
            for d in self.__iter__(self.random):
                yield d
# 对抗训练函数
def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数
# 模型训练
def do_train(mode = 'bert',filename = 'roberta', lastfour = False, LR = 1e-5, DR = 0.2,ext=False, batch_size=16):
    skf = StratifiedKFold(5, shuffle=True, random_state=2020)
    nfold = 1
    if (ext):
        data = np.concatenate([train_ext_data, valid_data],axis=0)
    else:
        data = np.concatenate([train_data,valid_data],axis=0)

    for train_index, valid_index in skf.split(data[:,:2], data[:,2:].astype('int')):
        train = data[train_index, :]
        valid = data[valid_index, :]

        train_generator = data_generator(train,batch_size)
        valid_generator = data_generator(valid, batch_size)

        model = build_model(mode = mode, filename=filename, lastfour = lastfour, LR=LR, DR = DR)

        adversarial_training(model,'Embedding-Token', 0.5)
        early_stopping = EarlyStopping(monitor='val_loss', patience=1,verbose = 1)

        if (ext):
            checkpoint = ModelCheckpoint(
                main_path + '/user_data/model_data/' + filename + '_weights/' + str(nfold) + '_ext.weights',
                monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1)
        else:
            checkpoint = ModelCheckpoint(main_path + '/user_data/model_data/' + filename + '_weights/' + str(nfold) + '.weights',
                                         monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1)

        model.fit_generator(train_generator.forfit(),
                            steps_per_epoch=len(train_generator),
                            epochs=5,
                            validation_data=valid_generator.forfit(),
                            validation_steps=len(valid_generator),
                            callbacks=[early_stopping, checkpoint],
                            verbose=2,
                            )
        del model
        K.clear_session()
        nfold += 1
# 运行
model = build_model(mode='bert', filename='bert', lastfour=False)
do_train(mode='bert', filename='bert', lastfour=False, LR=1e-5)
do_train(mode='bert', filename='bert', lastfour=False, LR=1e-5, ext=True)

model = build_model(mode='bert', filename='ernie', lastfour=False)
do_train(mode='bert', filename='ernie', lastfour=False, LR=1e-5)
do_train(mode='bert', filename='ernie', lastfour=False, LR=1e-5, ext=True)

model = build_model(mode='bert', filename='roberta', lastfour=False)
do_train(mode='bert', filename='roberta', lastfour=False, LR=1e-6, batch_size=8)





