import os
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from collections import defaultdict

# ========== 配置 ==========
CLIP_MODEL = "openai/clip-vit-base-patch32"
DATA_FILE = "twitter_dataset/devset/posts.txt"   # 格式：id\tlabel\ttext\timage_path
SAVE_DATA_PATH = "train.pkl"
SAVE_GRAPH_PATH = "graph_max_nodes_train.pkl"
IMAGE_DIR = 'twitter_dataset/devset/images'
DATA_AMOUNT = 100
label_dict = {'fake': 0, 'real': 1}

# ========== 模型初始化 ==========
clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
clip_model.eval()

# ========== 图像预处理 ==========
preprocess_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

def cosine_similarity(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)

def load_image(img_path):
    # 只返回PIL.Image，不做任何transform
    return Image.open(img_path).convert("RGB")

def extract_features(post, image_pil):
    # 直接用clip_model处理inputs
    inputs = clip_processor(text=post, images=image_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        Tk = outputs.text_model_output.last_hidden_state.squeeze(0)  # (m+1, d)
        Vk = outputs.vision_model_output.last_hidden_state.squeeze(0)  # (n+1, d)
    # 若维度不一致，做对齐
    d_text, d_img = Tk.shape[-1], Vk.shape[-1]
    if d_text != d_img:
        min_dim = min(d_text, d_img)
        Tk = Tk[:, :min_dim]
        Vk = Vk[:, :min_dim]
    return Tk, Vk

def dummy_ner(text_tokens):
    # 仅用于演示，用于模拟NER位置
    m = len(text_tokens)
    return min(1, m-1), min(2, m-1), min(3, m-1)  # (ts_idx, to_idx, tloc_idx)

def find_most_similar(Vk, target_vec):
    sims = [cosine_similarity(obj, target_vec) for obj in Vk[1:]]
    return sims.index(max(sims)) + 1  # +1 to skip CLS

def build_graph(Tk, Vk, ts_idx, to_idx, tloc_idx):
    Hk = torch.cat([Tk, Vk], dim=0)  # (m+n+2, d)
    m, n = Tk.shape[0] - 1, Vk.shape[0] - 1
    edges2others = []
    for i in range(Hk.shape[0]):
        for j in range(Hk.shape[0]):
            if i != j and cosine_similarity(Hk[i], Hk[j]) > 0.6:
                edges2others.append((i, j))

    # edges2postgraph
    edges2postgraph = [(0, ts_idx), (0, to_idx), (0, tloc_idx)]

    # edges2imagegraph
    vs_idx = find_most_similar(Vk, Tk[ts_idx]) + m + 1
    vo_idx = find_most_similar(Vk, Tk[to_idx]) + m + 1
    vloc_idx = find_most_similar(Vk, Tk[tloc_idx]) + m + 1
    edges2imagegraph = [(m + 1, vs_idx), (m + 1, vo_idx), (m + 1, vloc_idx)]

    type2nidxs = {
        "post_token": list(range(m + 1)),
        "image_obj": list(range(m + 1, Hk.shape[0])),
        "post_event": [ts_idx, to_idx, tloc_idx],
        "img_event": [vs_idx, vo_idx, vloc_idx],
        "cls_post": 0,
        "cls_img": m + 1
    }

    return {
        "nodes_features": Hk,
        "edges2postgraph": edges2postgraph,
        "edges2imagegraph": edges2imagegraph,
        "edges2others": edges2others,
        "edges2fcnodes": [],
        "type2nidxs": type2nidxs
    }

def process_dataset():
    data_list = []
    graph_list = []


    with open(DATA_FILE, "r", encoding="utf-8") as f:
        next(f)
        for idx, line in enumerate(f):
            if idx >= DATA_AMOUNT:
                break

            id_, text, _, image_ids, _, _, label = line.strip().split('\t')
            label = label_dict.get(label, -1)  # 转换标签为数字
            image_ids = image_ids.split(',')[0]  # 取第一个图片ID
            for type in ['jpg', 'png', 'jpeg','gif']:
                img_path = os.path.join(IMAGE_DIR, f"{image_ids}.{type}")
                if os.path.isfile(img_path):
                    break

            # Dummy token-level NER: assume tokens by whitespace
            tokens = text.strip().split()
            ts_idx, to_idx, tloc_idx = dummy_ner(tokens)

            image_tensor = load_image(img_path)
            Tk, Vk = extract_features(text, image_tensor)
            # Dummy token-level NER: assume tokens by whitespace
            tokens = text.strip().split()
            ts_idx, to_idx, tloc_idx = dummy_ner(tokens)

            # 保存data
            data_list.append({
                "id": id_,
                "post": Tk,
                "dct_img": Vk,
                "label": int(label)
            })

            # 构造图
            graph = build_graph(Tk, Vk, ts_idx, to_idx, tloc_idx)
            graph_list.append(tuple(graph.values()))

    # 保存
    for split in['train', 'eval', 'test']:
        with open(f"../dataset/{split}.pkl", 'wb') as f:
            pickle.dump(data_list, f)
        with open(f"../dataset/graph_max_nodes_{split}.pkl", 'wb') as f:
            pickle.dump(graph_list, f)

    print(f"Saved data to {SAVE_DATA_PATH} and graph to {SAVE_GRAPH_PATH}.")

if __name__ == '__main__':
    process_dataset()