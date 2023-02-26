# sentence_similarity
类别不平衡问题，也叫“长尾问题”，是机器学习面临的常见问题之一，尤其是来源于真实场景下的数据集，几乎都是类别不平衡的。</br>
利用语义相似度数据集，模拟真实场景下的数据，构造相似数据:不相似数据=1:4，解决类别不平衡问题。采用AUC（Area Under Curve）评估指标。
# 所需环境
Python==3.6</br>
tensorflow==1.14.0</br>
keras==2.3.1</br>
bert4keras==0.11.4</br>
笔者使用了开源的bert4keras，一个keras版的transformer模型库。bert4keras的更多介绍参见[这里](https://github.com/bojone/bert4keras)。
# 项目目录
├── bert4keras</br>
├── datasets    存放数据</br>
├── pretrained_model    存放预训练模型</br>
├── sentence_similarity_train.py    训练代码</br>
├── sentence_similarity_predict.py    评估和测试代码</br>
# 数据集
采用[哈工大LCQMC数据集](https://github.com/dreams-flying/NLP_Datasets)，对原始数据进行了筛选，处理好的数据存放在datasets/文件夹下。</br>

训练集和验证集中的数据统计情况：
| 数据集 | 相似 | 不相似 |
| :------:| :------: | :------: |
| train | 200 | 800 |
| dev | 50 | 200 |
# 使用说明
1.[下载预训练语言模型](https://github.com/ymcui/Chinese-BERT-wwm)</br>
  可采用chinese_roberta_wwm_ext等模型</br>
2.构建数据集(数据集已处理好)</br>
  LCQMC_train.json和LCQMC_dev.json</br>
3.训练模型
```
python sentence_similarity_train.py
```
4.评估和测试
```
python sentence_similarity_predict.py
```
# 结果
| 数据集 | acc | auc |
| :------:| :------: | :------: |
| dev | 0.8600 | 0.87340 |
# 参考
https://spaces.ac.cn/archives/7615
