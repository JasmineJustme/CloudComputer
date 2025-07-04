# -*- coding = utf-8 -*-
# @Time     : 2025/6/30 23:17
# @Author   : Yao Jiamin
# @File     : preproces.py
# @Software : PyCharm
import os
import pickle
import numpy as np
import torch
from pathlib import Path

TEXT_DIM = 822
IMG_DIM = 256
DIM_NODE_FEATURES = 512
label_dict = {'fake': 0, 'real': 1}

def extract_text_feature(text):
    # 实际应替换为BERT等模型
    return torch.tensor(np.random.rand(TEXT_DIM), dtype=torch.float32)

def extract_image_feature(img_path):
    # 实际应替换为ResNet等模型
    return torch.tensor(np.random.rand(IMG_DIM), dtype=torch.float32)

def build_graph_data(data):
    nodes_features = {}
    edges2postgraph = {}
    edges2imagegraph = {}
    edges2others = {}
    edges2fcnodes = {}
    type2nidxs = {}
    for d in data:
        Id = d['Id']
        # nodes_features: Tensor
        feat = torch.cat([d['post'], d['dct_img']])
        if feat.shape[0] < DIM_NODE_FEATURES:
            pad = torch.zeros(DIM_NODE_FEATURES - feat.shape[0])
            feat = torch.cat([feat, pad])
        else:
            feat = feat[:DIM_NODE_FEATURES]
        nodes_features[Id] = feat.unsqueeze(0)
        # 边和权重都用Tensor
        edges2postgraph[Id] = {'index': torch.tensor([[0],[0]], dtype=torch.long), 'weight': torch.tensor([[1.0]], dtype=torch.float32)}
        edges2imagegraph[Id] = {'index': torch.tensor([[0],[0]], dtype=torch.long), 'weight': torch.tensor([[1.0]], dtype=torch.float32)}
        edges2others[Id] = {'index': torch.tensor([[0],[0]], dtype=torch.long), 'weight': torch.tensor([[1.0]], dtype=torch.float32)}
        edges2fcnodes[Id] = {'index': torch.tensor([[0],[0]], dtype=torch.long), 'weight': torch.tensor([[1.0]], dtype=torch.float32)}
        type2nidxs[Id] = {'post_subgraph': [0], 'image_subgraph': [0], 'all_nodes': [0]}
    return (nodes_features, edges2postgraph, edges2imagegraph, edges2others, edges2fcnodes, type2nidxs)

def main(posts_txt, image_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    data = []
    with open(posts_txt, 'r', encoding='utf-8') as f:
        next(f)
        for idx, line in enumerate(f):
            if idx >= data_amount:
                break
            line = line.strip().split('\t')
            if not line:
                continue
            try:
                post_id, text, _, image_ids, _, _, label = line
                image_ids = image_ids.split(',')[0]
            except ValueError:
                print(f"格式错误：{line}")
                continue
            img_path = os.path.join(image_dir, f"{image_ids}.jpg")
            if not os.path.isfile(img_path):
                print(f"图片缺失：{post_id}:{img_path}")
                continue
            text_feat = extract_text_feature(text)
            img_feat = extract_image_feature(img_path)
            data.append({
                "Id": post_id,
                "post": text_feat,         # torch.Tensor
                "dct_img": img_feat,       # torch.Tensor
                "label": int(label_dict[label]),  # int
            })
    graph_data = build_graph_data(data)
    # 保存train.pkl
    with open(output_dir / 'train.pkl', 'wb') as f:
        pickle.dump(data, f)
    # 构建并保存graph_max_nodes_train.pkl
    with open(output_dir / 'graph_max_nodes_train.pkl', 'wb') as f:
        pickle.dump(graph_data, f)
    print(f"已保存 {len(data)} 条数据到 {output_dir}")

    # 保存dev.pkl
    with open(output_dir / 'dev.pkl', 'wb') as f:
        pickle.dump(data, f)
    # 构建并保存graph_max_nodes_dev.pkl
    with open(output_dir / 'graph_max_nodes_dev.pkl', 'wb') as f:
        pickle.dump(graph_data, f)
    print(f"已保存 {len(data)} 条数据到 {output_dir}")

    # 保存test.pkl
    with open(output_dir / 'test.pkl', 'wb') as f:
        pickle.dump(data, f)
    # 构建并保存graph_max_nodes_test.pkl
    with open(output_dir / 'graph_max_nodes_test.pkl', 'wb') as f:
        pickle.dump(graph_data, f)
    print(f"已保存 {len(data)} 条数据到 {output_dir}")

if __name__ == '__main__':
    data_amount = 100
    posts_txt = 'twitter_dataset/devset/posts.txt'
    image_dir = 'twitter_dataset/devset/images'
    output_dir = '../dataset'
    main(posts_txt, image_dir, output_dir)