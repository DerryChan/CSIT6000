import os
import sys
import jieba
import pandas as pd
import codecs
import math
import random

stopwords_set = set()
basedir = '/Users/derry/Desktop/HKUST/Introduction to big data/Sohu Competition/CSIT6000/Data/'

# 分词结果文件
train_file = codecs.open(basedir + "news.data.seg.train", 'w', 'utf-8')
test_file = codecs.open(basedir + "news.data.seg.test", 'w', 'utf-8')

# 停用词文件
with open(basedir + 'stop_text.txt', 'r', encoding='utf-8') as infile:
    for line in infile:
        stopwords_set.add(line.strip())

train_data = pd.read_table(basedir + 'News_info_train.txt', header=None, error_bad_lines=False)
label_data = pd.read_table(basedir + 'News_pic_label_train.txt', header=None, error_bad_lines=False)

train_data.drop([2], axis=1, inplace=True)
train_data.columns = ['id', 'text']
label_data.drop([2, 3], axis=1, inplace=True)
label_data.columns = ['id', 'class']
train_data = pd.merge(train_data, label_data, on='id', how='outer')

for index, row in train_data.iterrows():
    # 结巴分词
    seg_text = jieba.cut(row['text'].replace("\t", " ").replace("\n", " "))
    outline = " ".join(seg_text)
    outline = " ".join(outline.split())

    # 去停用词与HTML标签
    outline_list = outline.split(" ")
    outline_list_filter = [item for item in outline_list if item not in stopwords_set]
    outline = " ".join(outline_list_filter)

    # 写入
    if not math.isnan(row['class']):
        outline = outline + "\t__label__" + str(int(row['class'])) + "\n"
        train_file.write(outline)
        train_file.flush()
        # if random.random() > 0.7:
        #     test_file.write(outline)
        #     test_file.flush()
        # else:
        #     train_file.write(outline)
        #     train_file.flush()

train_file.close()
test_file.close()