import jieba
import pandas as pd
import codecs

stopwords_set = set()
basedir = '/Users/derry/Desktop/HKUST/Introduction to big data/Sohu Competition/CSIT6000/Data/'

# 分词结果文件
validate_file = codecs.open(basedir + "news.data.seg.validate", 'w', 'utf-8')

# 停用词文件
with open(basedir + 'stop_text.txt', 'r', encoding='utf-8') as infile:
    for line in infile:
        stopwords_set.add(line.strip())

validate_data = pd.read_table(basedir + 'News_info_validate.txt', header=None, error_bad_lines=False)

validate_data.drop([2], axis=1, inplace=True)
validate_data.columns = ['id', 'text']

for index, row in validate_data.iterrows():
    # 结巴分词
    seg_text = jieba.cut(row['text'].replace("\t", " ").replace("\n", " "))
    outline = " ".join(seg_text)
    outline = " ".join(outline.split())

    # 去停用词与HTML标签
    outline_list = outline.split(" ")
    outline_list_filter = [item for item in outline_list if item not in stopwords_set]
    outline = " ".join(outline_list_filter)

    # 写入
    outline = outline + "\n"
    validate_file.write(outline)
    validate_file.flush()

validate_file.close()
