import os
import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import random
import argparse
import time
import sys

sys.path.append('Embeddings/embeddings/resources/TA_TransE')

from utils import *
from data import *
import model as model
import loss as loss

USE_CUDA = torch.cuda.is_available()
longTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
floatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class Config(object):
    def __init__(self):
        self.dropout = 0
        self.dataset = None
        self.learning_rate = 1e-3
        self.early_stopping_round = 0
        self.L1_flag = True
        self.embedding_size = 300  #300
        self.train_times = 200    #750
        self.margin = 5.0
        self.filter = True
        self.momentum = 0.9
        self.optimizer = optim.Adam
        self.loss_function = loss.marginLoss()
        self.loss_type = 0
        self.entity_total = 0
        self.relation_total = 0
        self.batch_size = 512

def load_targets(target_file):
    with open(target_file, "r") as f:
        return set(line.strip() for line in f)

def extract_embeddings(entity_list, entity2id_path, embeddings, output_path, experiment_name):
    with open(entity2id_path, 'r') as f:
        lines = f.readlines()[1:]  # skip header
        entity2id = dict(line.strip().split('\t') for line in lines)

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{experiment_name}_embeddings.txt")
    with open(output_file, 'w') as f:
        match_count = 0
        for entity in entity_list:
            if entity in entity2id:
                eid = int(entity2id[entity])
                vector = "\t".join(map(str, embeddings[eid]))
                f.write(f"{entity}\t{vector}\n")
                match_count += 1
    print(f"✅ Saved {match_count} entity embeddings to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--target_entity_file', type=str, required=True)
    parser.add_argument('--entity2id_file', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--train_times', type=int, default=750)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()

    # === Load Data ===
    print(f'🎯 Loading the Data ... ')
    data_path = os.path.join(args.datapath, args.dataset + '_TA')
    temporal_list = np.load(os.path.join(data_path, 'train_tem.npy')).tolist()

    train_total, train_list, train_dict, _ = load_quadruples(data_path, 'train2id.txt', 'train_tem.npy')
    entity_total, relation_total, _ = get_total_number(data_path, 'stat.txt')
    quadTotal, qualList, tripleDict, _ = load_quadruples(
        data_path,
        'train2id.txt', 'train_tem.npy',
        'valid2id.txt', 'valid_tem.npy',
        'test2id.txt', 'test_tem.npy'
    )

    config = Config()
    config.dataset = args.dataset
    config.embedding_size = args.embedding_size
    config.train_times = args.train_times
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.entity_total = entity_total
    config.relation_total = relation_total

    device = torch.device("cuda" if USE_CUDA else "cpu")
    model = model.TATransEModel(config).to(device)
    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
    margin = torch.tensor(config.margin).to(device)
    # if USE_CUDA:
    #     model.cuda()
    # optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
    # loss_function = config.loss_function()
    # if USE_CUDA:
    #     loss_function.cuda()
    # margin = autograd.Variable(floatTensor([config.margin]))

    # ====== TRAIN =====
    print("🎯 Starting training...")
    train_batches = getBatchList(train_list, config.batch_size)
    print("✅ Batching completed.")
    print("Total number of batches:", len(train_batches))
    print("First batch size:", len(train_batches[0]) if train_batches else "No batches")

    for epoch in range(config.train_times):
        model.train()
        total_loss = floatTensor([0.0]).to(device)
        random.shuffle(train_batches)
        start = time.time()
        for i, batch in enumerate(train_batches):
            batch_start = time.time()

            if config.filter:
                pos_h, pos_t, pos_r, pos_tem, neg_h, neg_t, neg_r, neg_tem = getBatch_filter_all(
                    batch, 
                    config.entity_total, 
                    tripleDict)
            else:
                pos_h, pos_t, pos_r, pos_tem, neg_h, neg_t, neg_r, neg_tem = getBatch_raw_all(
                    batch, 
                    config.entity_total)

            pos_h = autograd.Variable(longTensor(pos_h))
            pos_t = autograd.Variable(longTensor(pos_t))
            pos_r = autograd.Variable(longTensor(pos_r))
            pos_tem = autograd.Variable(longTensor(pos_tem))
            neg_h = autograd.Variable(longTensor(neg_h))
            neg_t = autograd.Variable(longTensor(neg_t))
            neg_r = autograd.Variable(longTensor(neg_r))
            neg_tem = autograd.Variable(longTensor(neg_tem))

            model.zero_grad()
            pos_score, neg_score = model(pos_h, pos_t, pos_r, pos_tem, neg_h, neg_t, neg_r, neg_tem)
            losses = config.loss_function(pos_score, neg_score, margin)
            ent_embeddings = model.ent_embeddings(torch.cat([pos_h, pos_t, neg_h, neg_t]))
            rseq_embeddings = model.get_rseq(torch.cat([pos_r, neg_r]), torch.cat([pos_tem, neg_tem]))
            losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rseq_embeddings)
            losses.backward()
            optimizer.step()
            total_loss += losses.data
            
            if i % 100 == 0:
                print(f"🌀 Epoch {epoch}, Batch {i}/{len(train_batches)}, Loss: {losses.item():.4f}, Time per batch: {(time.time() - batch_start):.2f}s", flush=True)
        
        print(f"✅ Epoch {epoch}: Loss = {total_loss[0]}")

    # # === Extract Embeddings ===
    print("📦 Extracting embeddings...")
    model.eval()
    entity_embeddings = model.ent_embeddings.weight.data.cpu().numpy()
    target_entities = load_targets(args.target_entity_file)

    extract_embeddings(
        entity_list=target_entities,
        entity2id_path=args.entity2id_file,
        embeddings=entity_embeddings,
        output_path=args.output_path,
        experiment_name=args.experiment
    )
