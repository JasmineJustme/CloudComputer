# -*- coding = utf-8 -*-
# @Time     : 2025/6/27 16:22
# @Author   : Yao Jiamin
# @File     : main2.py
# @Software : PyCharm
import re
import nltk
import gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import numpy as np
import seaborn as sns


# 第一次运行需要下载资源
# nltk.download('stopwords')
# nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def load_news_data():
    global news_data
    with open('twitter_dataset/devset/posts.txt', 'r', encoding='utf-8') as file:
        next(file)
        for line in file:
            # 分割每行数据（假设数据以制表符分隔）
            parts = line.strip().split('\t')

            # 提取字段
            post_id = parts[0]
            post_text = parts[1].split('http')[0]
            user_id = parts[2]
            image_ids = parts[3]
            username = parts[4]
            timestamp = parts[5]
            label = parts[6]

            # 将数据存储为字典
            post_data = {
                'post_id': post_id,
                'post_text': post_text,
                'user_id': user_id,
                'image_ids': image_ids,
                'username': username,
                'timestamp': timestamp,
                'label': label
            }
            # 添加到列表
            news_data.append(post_data)

def data_preprocess(doc):
    # 去除非字母字符，转小写，分词
    words = re.sub(r'[^a-zA-Z]', ' ', doc).lower().split()
    # 去除停用词和短词
    words = [w for w in words if w not in stop_words and len(w) > 2]
    # 词形还原
    words = [lemmatizer.lemmatize(w) for w in words]
    return words

def LDA_train():
    # 打印主题关键词
    for idx, topic in lda_model.print_topics(-1):
        print(f"主题 #{idx}:\n{topic}\n")

def visualize_analysis():
    # （1）pyLDAvis 可视化
    # pyLDAvis.enable_notebook()
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'lda_vis.html')  # 输出为HTML文件

    # （2）词云图：每个主题的关键词
    for i in range(lda_model.num_topics):
        plt.figure()
        plt.imshow(WordCloud(background_color='white').fit_words(dict(lda_model.show_topic(i, 30))))
        plt.axis("off")
        plt.title(f"Topic #{i}")
        plt.show()

    # （3）（可选）热力图展示文档-主题分布矩阵

    doc_topic_matrix = np.array([
        [topic_prob for _, topic_prob in lda_model.get_document_topics(doc, minimum_probability=0.0)]
        for doc in corpus
    ])

    plt.figure(figsize=(10, 6))
    sns.heatmap(doc_topic_matrix, annot=False, cmap='Blues')
    plt.xlabel("Topic")
    plt.ylabel("Document")
    plt.title("Document-Topic Distribution Heatmap")
    plt.show()

if __name__ == '__main__':
    news_data = []

    load_news_data()
    processed_docs = [data_preprocess(news['post_text']) for news in news_data]
    # 创建词典（词-ID 映射）
    dictionary = corpora.Dictionary(processed_docs)
    # 转换为词袋模型语料库
    corpus = [dictionary.doc2bow(text) for text in processed_docs]
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=3,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    visualize_analysis()

