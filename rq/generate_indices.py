import collections
import json
import logging
import argparse

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE

import os
import sys
import glob

# 添加 rq 目录到 Python 路径以支持相对导入
RQ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RQ_DIR)

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description='Generate SID indices from RQ-VAE model')
    parser.add_argument('--dataset', type=str, default='Industrial_and_Scientific',
                        help='Dataset name')
    parser.add_argument('--rqvae_dir', type=str, required=True,
                        help='Directory containing RQ-VAE checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to data_root)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    dataset = args.dataset

    # 查找 best_collision_model.pth 文件
    ckpt_candidates = glob.glob(os.path.join(args.rqvae_dir, '**/best_collision_model.pth'), recursive=True)
    if not ckpt_candidates:
        # 如果没找到,尝试找其他 .pth 文件
        ckpt_candidates = glob.glob(os.path.join(args.rqvae_dir, '**/*.pth'), recursive=True)

    if not ckpt_candidates:
        raise FileNotFoundError(f"No checkpoint found in {args.rqvae_dir}")

    ckpt_path = ckpt_candidates[0]  # 使用第一个找到的检查点
    print(f"Using checkpoint: {ckpt_path}")

    output_dir = args.output_dir or args.data_root
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{dataset}.index.json"
    output_file = os.path.join(output_dir, output_file)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt = torch.load(
        ckpt_path,
        map_location=torch.device('cpu'),
        weights_only=False
    )

    model_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    # 使用训练时的数据路径或者从 data_root 推导
    data_path = os.path.join(args.data_root, f"{dataset}.emb.npy")
    if not os.path.exists(data_path):
        # 尝试使用模型训练时的路径
        if hasattr(model_args, 'data_path') and os.path.exists(model_args.data_path):
            data_path = model_args.data_path
        else:
            raise FileNotFoundError(f"Embedding file not found at {data_path}")

    print(f"Loading embeddings from: {data_path}")
    data = EmbDataset(data_path)

    model = RQVAE(in_dim=data.dim,
                      num_emb_list=model_args.num_emb_list,
                      e_dim=model_args.e_dim,
                      layers=model_args.layers,
                      dropout_prob=model_args.dropout_prob,
                      bn=model_args.bn,
                      loss_type=model_args.loss_type,
                      quant_loss_weight=model_args.quant_loss_weight,
                      kmeans_init=model_args.kmeans_init,
                      kmeans_iters=model_args.kmeans_iters,
                      sk_epsilons=model_args.sk_epsilons,
                      sk_iters=model_args.sk_iters,
                      )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    num_workers = getattr(model_args, 'num_workers', 4)
    data_loader = DataLoader(data, num_workers=num_workers,
                                 batch_size=64, shuffle=False,
                                 pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d,use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices.append(code)
            all_indices_str.append(str(code))
        # break

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon=0.0
    # model.rq.vq_layers[-1].sk_epsilon = 0.005
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0
    #There are often duplicate items in the dataset, and we no longer differentiate them
    while True:
        if tt >= 20 or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(collision_item_groups)
        print(len(collision_item_groups))
        for collision_items in collision_item_groups:
                d = data[collision_items].to(device)

                indices = model.get_indices(d, use_sk=True)
                indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                for item, index in zip(collision_items, indices):
                    code = []
                    for i, ind in enumerate(index):
                        code.append(prefix[i].format(int(ind)))

                    all_indices[item] = code
                    all_indices_str[item] = str(code)
        tt += 1


    print("All indices number: ",len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate",(tot_item-tot_indice)/tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)



    with open(output_file, 'w') as fp:
        json.dump(all_indices_dict,fp)

    print(f"✓ Index file saved to: {output_file}")
