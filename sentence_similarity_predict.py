#! -*- coding:utf-8 -*-
# 句子对分类任务，LCQMC数据集
# val_acc: 0.8600, auc=0.87340
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1
import json
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense

set_gelu('tanh')  # 切换gelu版本

maxlen = 128
batch_size = 64

config_path = 'pretrained_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'pretrained_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'pretrained_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D

def load_data1(filename):
    D = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            l = json.loads(line)
            D.append((l["text1"], l["text2"], int(l["label"])))
    return D



# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []



# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False,
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(
    units=2, kernel_initializer=bert.initializer
)(output)#, activation='softmax'

model = keras.models.Model(bert.model.input, output)
# model.summary()



def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total

def evaluate1(data):
    from sklearn.metrics import roc_auc_score
    y_true = []
    y_score = []
    for x_true, y_true_list in data:
        y_pred = model.predict(x_true)
        for y_t, y_p in zip(y_true_list, y_pred):
            y_true.append(y_t[0])
            y_score.append(y_p[1])

    auc = roc_auc_score(y_true, y_score)
    return auc


if __name__ == '__main__':

    model.load_weights('./save/best_model.weights')

    # 加载数据集
    # train_data = load_data1('datasets/LCQMC_train.json')
    valid_data = load_data1('datasets/LCQMC_dev.json')

    # print("train_data", len(train_data))
    print("valid_data", len(valid_data))


    # 转换数据集
    valid_generator = data_generator(valid_data, batch_size)
    val_acc = evaluate(valid_generator)

    # print(
    #     u'val_acc: %.5f, test_acc: %.5f\n' % (val_acc, test_acc)
    # )

    print(u'val_acc: %.5f\n' % (val_acc))