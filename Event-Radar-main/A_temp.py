# -*- coding = utf-8 -*-
# @Time     : 2025/7/3 22:22
# @Author   : Yao Jiamin
# @File     : A_temp.py
# @Software : PyCharm
import os
import pickle
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import stanza

# NER工具初始化
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

def extract_entities(text):
    doc = nlp(text)
    entities = {
        "subject": [],
        "object": [],
        "loc": [],
        "tokens": []
    }
    for sent in doc.sentences:
        for token in sent.tokens:
            entities["tokens"].append(token.text)
        for ent in sent.ents:
            if ent.type in ['PERSON', 'ORG']:
                entities["subject"].append(ent.text)
            elif ent.type in ['LOC', 'GPE']:
                entities["loc"].append(ent.text)
            elif ent.type in ['PRODUCT', 'OBJECT']:
                entities["object"].append(ent.text)
    return entities

# Faster R-CNN目标检测
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_fasterrcnn_model(num_classes=91):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.eval()
    return model

fasterrcnn_model = get_fasterrcnn_model()
fasterrcnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_image_objects(image_path, max_objects=5, feature_dim=512):
    if not os.path.exists(image_path):
        return np.zeros((1, feature_dim))
    image = Image.open(image_path).convert('RGB')
    img_tensor = fasterrcnn_transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = fasterrcnn_model(img_tensor)[0]
    features = []
    scores = predictions['scores'].cpu().numpy() if 'scores' in predictions else []
    boxes = predictions['boxes'].cpu().numpy() if 'boxes' in predictions else []
    order = np.argsort(-scores)[:max_objects] if len(scores) > 0 else []
    for idx in order:
        box = boxes[idx]
        x1, y1, x2, y2 = map(int, box)
        region = image.crop((x1, y1, x2, y2)).resize((32, 32))
        region_feat = np.array(region).reshape(-1)
        if region_feat.shape[0] < feature_dim:
            region_feat = np.concatenate([region_feat, np.zeros(feature_dim - region_feat.shape[0])])
        else:
            region_feat = region_feat[:feature_dim]
        features.append(region_feat)
    if len(features) == 0:
        features.append(np.zeros(feature_dim))
    features = np.stack(features, axis=0)
    return features

def compute_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def parse_posts_txt(posts_txt_path, images_dir):
    """
    解析posts.txt，每行为一条，返回list[dict]，字典键包括：Id, post, dct_img, label
    """
    import csv
    results = []
    label_dict = {'fake': 0, 'real': 1}
    with open(posts_txt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            post_id = row.get('post_id') or row.get('postid') or row.get('id')
            post_text = row.get('post_text') or row.get('text') or row.get('post')
            image_id = row.get('image_id') or row.get('imageid') or row.get('image') or row.get('image_id(s)')
            label = row.get('label')
            label = label_dict.get(label, -1)  # 转换标签为数字
            image_ids = image_ids.split(',')[0]  # 取第一个图片ID
            for type in ['jpg', 'png', 'jpeg', 'gif']:
                img_fp = os.path.join(images_dir, f"{image_id}.{type}")
                if os.path.isfile(img_fp):
                    break
            results.append({
                "Id": post_id,
                "post": post_text,
                "dct_img": img_fp,
                "label": label
            })
    return results

def build_graph_data(instances, clip_model, clip_processor, embedding_dim, text_size, dim_node_features):
    nodes_features = {}
    edges2postgraph = {}
    edges2imagegraph = {}
    edges2others = {}
    edges2fcnodes = {}
    type2nidxs = {}

    for idx, instance in enumerate(tqdm(instances)):
        Id = instance["Id"]
        text = instance["post"]
        image_path = instance["dct_img"]

        # 文本特征
        entities = extract_entities(text)
        tokens = entities["tokens"]
        text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=text_size)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs).cpu().numpy()[0]
        # 图像特征
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image_inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = clip_model.get_image_features(**image_inputs).cpu().numpy()[0]
            object_features = extract_image_objects(image_path, max_objects=5, feature_dim=dim_node_features)
        else:
            image_features = np.zeros(embedding_dim)
            object_features = np.zeros((1, dim_node_features))

        # 节点特征拼接
        node_features = np.concatenate([text_features[:embedding_dim][np.newaxis, :], object_features[:, :embedding_dim]], axis=0)
        nodes_features[Id] = node_features

        # 边权
        num_nodes = node_features.shape[0]
        edge_weights = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    sim = compute_similarity(node_features[i], node_features[j])
                    edge_weights[i, j] = (sim + 1) / 2

        edges2postgraph[Id] = edge_weights
        edges2imagegraph[Id] = edge_weights
        edges2others[Id] = edge_weights
        edges2fcnodes[Id] = edge_weights
        type2nidxs[Id] = {
            "subject": [0],
            "object": [1] if num_nodes > 1 else [],
            "loc": [2] if num_nodes > 2 else [],
            "tokens": list(range(min(len(tokens), num_nodes)))
        }

    graph_data = {
        "nodes_features": nodes_features,
        "edges2postgraph": edges2postgraph,
        "edges2imagegraph": edges2imagegraph,
        "edges2others": edges2others,
        "edges2fcnodes": edges2fcnodes,
        "type2nidxs": type2nidxs
    }
    return graph_data

def main():
    # 数据路径
    root_dir = "twitter_dataset/devset"
    posts_txt_path = os.path.join(root_dir, "posts.txt")
    images_dir = os.path.join(root_dir, "images")

    # 解析原始数据
    instances = parse_posts_txt(posts_txt_path, images_dir)

    # CLIP参数
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    embedding_dim = 64
    text_size = 822
    dim_node_features = 512

    graph_data = build_graph_data(instances, clip_model, clip_processor, embedding_dim, text_size, dim_node_features)
    with open(os.path.join(root_dir, "graph_data.pkl"), "wb") as f:
        pickle.dump(graph_data, f)

    dct_data = []
    for instance in instances:
        dct_data.append({
            "label": instance["label"],
            "Id": instance["Id"],
            "dct_img": instance["dct_img"],
            "post": instance["post"]
        })
    with open(os.path.join(root_dir, "dct_data.pkl"), "wb") as f:
        pickle.dump(dct_data, f)

if __name__ == "__main__":
    main()