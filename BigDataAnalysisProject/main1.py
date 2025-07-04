# -*- coding = utf-8 -*-
# @Time     : 2025/6/26 14:17
# @Author   : Yao Jiamin
# @File     : main1.py
# @Software : PyCharm

import ollama
import time
import re


news_data = []
sleep_time = 0  # 设置每次请求之间的延时，单位为秒

def load_news_data():
    global news_data
    with open('twitter_dataset/devset/posts.txt', 'r', encoding='utf-8') as file:
        next(file)
        for line in file:
            # 分割每行数据（假设数据以制表符分隔）
            parts = line.strip().split('\t')

            # 提取字段
            post_id = parts[0]
            post_text = parts[1]
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

def ask_chatglm3(prompt):
    response = ollama.chat(model='EntropyYue/chatglm3', messages=[{"role": "user", "content": prompt,"temperature": 0.01}])
    time.sleep(sleep_time)
    return response['message']['content']

# 1. 判别真假
def assignment1():
    print("Assignment1开始判断新闻真假...")
    fake_correct_prediction = 0
    true_correct_prediction = 0
    fake_count_actual = 0
    true_count_actual = 0
    rounds = 0
    posts_len = len(news_data)
    for news in news_data:
        rounds += 1
        recurison_count = 0
        while True:
            prompt = f'''请判断：以后的新闻是真新闻还是假新闻，：\n{news['post_text'].split('http')[0]}\n 并严格将答案按照以下格式输出“判断结果：fake”或“判断结果：true”。'''
            result = ask_chatglm3(prompt).strip().lower()
            recurison_count += 1
            if 'fake' in result or 'true' in result or recurison_count > 10:
                break
        # print(result)
        if news['label'] == 'fake':
            fake_count_actual += 1
            fake_correct_prediction += 1 if 'fake' in result else 0
        else:
            true_count_actual += 1
            true_correct_prediction += 1 if 'true' in result else 0
        if rounds / posts_len * 100 // 10 - (rounds-1) / posts_len * 100 // 10 == 1:  # 每10%输出一次进度:
            print(f"已处理{rounds}条新闻数据，当前进度：{rounds / posts_len * 100:.2f}，"
                  f"比率为{(fake_correct_prediction+true_correct_prediction) / (fake_count_actual+true_count_actual)*100:.2f}%")
    accuracy1 = (fake_correct_prediction+true_correct_prediction) / len(news_data)
    accuracy2 = fake_correct_prediction / fake_count_actual if fake_count_actual > 0 else 0
    accuracy3 = true_correct_prediction / true_count_actual if true_count_actual > 0 else 0
    return [accuracy1, accuracy2, accuracy3]

def assignment2():
    # 2. 情感分析
    for news in news_data:
        recurison_count = 0
        while True:
            prompt = f'''请判断：以后的新闻是真新闻还是假新闻：\n{news['post_text'].split('http')[0]}\n 并严格将答案按照以下格式输出“判断结果：positive”或“判断结果：neutral”或“判断结果：negative”。'''
            result = ask_chatglm3(prompt).strip().lower()
            recurison_count += 1
            if 'positive' in result or 'neutral' in result or 'negative' in result or recurison_count > 10:
                match = re.search(r'(positive|neutral|negative)', result)
                news['sentiment'] = match.group(1) if match else ''
                break

def assignment3():
    print("Assignment3开始判断新闻真假...")
    fake_correct_prediction = 0
    true_correct_prediction = 0
    fake_count_actual = 0
    true_count_actual = 0
    rounds = 0
    posts_len = len(news_data)
    for news in news_data:
        rounds += 1
        recurison_count = 0
        while True:
            prompt = f'''请结合情感分析类别：{news['sentiment']}，判断以下新闻是真新闻还是假新闻：\n{news['post_text'].split('http')[0]}\n 
            并严格将答案按照以下格式输出“判断结果：fake”或“判断结果：true”。'''
            result = ask_chatglm3(prompt).strip().lower()
            recurison_count += 1
            if 'fake' in result or 'true' in result or recurison_count > 10:
                match = re.search(r'(fake|true)', result)
                news['sentiment'] = match.group(1) if match else ''
                break
        if news['label'] == 'fake':
            fake_count_actual += 1
            fake_correct_prediction += 1 if 'fake' in result else 0
        else:
            true_count_actual += 1
            true_correct_prediction += 1 if 'true' in result else 0
        if rounds / posts_len * 100 // 10 - (rounds-1) / posts_len * 100 // 10 == 1:  # 每10%输出一次进度:
            print(f"已处理{rounds}条新闻数据，当前进度：{rounds / posts_len * 100:.2f}，"
                  f"比率为{(fake_correct_prediction+true_correct_prediction) / (fake_count_actual+true_count_actual)*100:.2f}%")
    accuracy1 = (fake_correct_prediction+true_correct_prediction) / len(news_data)
    accuracy2 = fake_correct_prediction / fake_count_actual if fake_count_actual > 0 else 0
    accuracy3 = true_correct_prediction / true_count_actual if true_count_actual > 0 else 0
    return [accuracy1, accuracy2, accuracy3]

def compare_accuracy(acc1, acc3):
    print("🔍 准确率对比：")
    print(f"原始真假判断准确率：{acc1:.2f}")
    print(f"结合情感分析准确率：{acc3:.2f}")
    for a,b in zip(acc1, acc3):
        if a > b:
            print("加入情感分析提升了判断准确率。")
        elif a == b:
            print("加入情感分析后准确率无明显变化。")
        else:
            print("加入情感分析后准确率反而下降。")

if __name__ == '__main__':
    load_news_data()
    # result1 = assignment1()
    # print(result1)
    # 其他任务可以在这里调用
    assignment2()
    result3 = assignment3()
    print(result3)
    # compare_accuracy(result1, result3)
