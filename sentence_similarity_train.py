#! -*- coding:utf-8 -*-
# 句子对分类任务，LCQMC数据集
# val_acc: 0.8600, auc=0.87340
import json
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
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

# 加载数据集
train_data = load_data1('datasets/LCQMC_train.json')
valid_data = load_data1('datasets/LCQMC_dev.json')
# test_data = load_data('datasets/LCQMC/test.txt')

print("train_data", len(train_data))
print("valid_data", len(valid_data))
# print("test_data", len(test_data))

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


def categorical_crossentropy_with_prior(y_true, y_pred, tau=1.0):
    """带先验分布的交叉熵
    注：y_pred不用加softmax
    """
    prior = [2]  # 自己定义好prior，shape为[num_classes]
    log_prior = K.constant(np.log(prior))# + 1e-8
    for _ in range(K.ndim(y_pred) - 1):
        log_prior = K.expand_dims(log_prior, 0)
    y_pred = y_pred + tau * log_prior
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def sparse_categorical_crossentropy_with_prior(y_true, y_pred, tau=1.0):
    """带先验分布的稀疏交叉熵
    注：y_pred不用加softmax
    """
    prior = [2]  # 自己定义好prior，shape为[num_classes]
    log_prior = K.constant(np.log(prior))# + 1e-8
    for _ in range(K.ndim(y_pred) - 1):
        log_prior = K.expand_dims(log_prior, 0)
    y_pred = y_pred + tau * log_prior
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


model.compile(
    loss=sparse_categorical_crossentropy_with_prior,
    # loss=categorical_crossentropy_with_prior,
    # loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
    # weighted_metrics=["accuracy"]
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
# test_generator = data_generator(test_data, batch_size)


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


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate1(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('save/best_model.weights')
        # test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    # from sklearn.utils.class_weight import compute_class_weight
    #
    # y_train = []
    # with open("datasets/LCQMC_train.json", encoding="utf-8") as f:
    #     for line in f:
    #         l = json.loads(line)
    #         y_train.append(int(l["label"]))
    #
    # class_weights = compute_class_weight("balanced", np.unique(y_train), y_train)
    # class_weights = dict(enumerate(class_weights))
    # print(class_weights)

    #处理非平衡训练数据，使得损失函数对样本数不足的数据更加关注，可以采用sklearn中的compute_class_weight，也可以自定义
    class_weights = {0: 0.5, 1: 1.0}#类别为0和1

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20,
        callbacks=[evaluator],
        class_weight = class_weights
    )

    # model.load_weights('best_model.weights')
    # print(u'final test acc: %05f\n' % (evaluate(test_generator)))

else:

    model.load_weights('best_model.weights')