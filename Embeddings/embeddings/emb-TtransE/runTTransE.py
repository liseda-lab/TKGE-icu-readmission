import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import random
import time
import argparse
import numpy as np
import sys

sys.path.append('/home/rcarvalho/Embeddings/embeddings/resources/TA_TransE')

from utils import *
from data import *
import model as model
import loss as loss

USE_CUDA = torch.cuda.is_available()
longTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
floatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def load_targets(target_file):
    with open(target_file, "r") as f:
        targets = [line.strip() for line in f]
    return set(targets)


def extract_embeddings_for_specific_entities(entity_list, entity2id_path, ent_embeddings, outPath, experimentName):
    entity2id = {}
    with open(entity2id_path, 'r') as f:
        for line in f.readlines()[1:]:  # Skip the first line (count line)
            try:
                entity, eid = line.strip().split('\t')
                entity2id[entity] = int(eid)
            except ValueError:
                continue

    out_file_path = os.path.join(outPath, f'{experimentName}_embeddings.txt')
    os.makedirs(outPath, exist_ok=True)

    with open(out_file_path, 'w') as f:
        match_count = 0
        for entity in entity_list:
            if entity in entity2id:
                eid = entity2id[entity]
                embedding = "\t".join(map(str, ent_embeddings[eid]))
                f.write(f"{entity}\t{embedding}\n")
                match_count += 1

    print(f"✅ Saved {match_count} matched entity embeddings to: {out_file_path}")

class Config(object):
    def __init__(self):
        self.dropout = 0
        self.dataset = None
        self.learning_rate = 1e-3
        self.early_stopping_round = 0
        self.L1_flag = True
        self.embedding_size = 300
        self.train_times = 300        #CHANGED FROM 1000
        self.margin = 5.0
        self.filter = True
        self.momentum = 0.9
        self.optimizer = optim.Adam          
        self.loss_function = loss.marginLoss()
        self.loss_type = 0
        self.entity_total = 0
        self.relation_total = 0
        self.batch_size = 512
        self.tem_total = 32

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--entity', type=str, required=True)
    parser.add_argument('--train_times', type=int, default=750)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--target_entity_file', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--experiment', type=str, required=True)
    args = parser.parse_args()

    # Load data
    print(f'🎯 Loading the Data ... ')
    data_path = f'{args.datapath}/{args.dataset}_TTransE'
    trainTotal, trainList, trainDict = load_quadruples_TTransE(data_path, 'train2id.txt')
    entity_total, relation_total, tem_total = get_total_number(data_path, 'stat.txt')
    
    config = Config()
    config.dataset = args.dataset
    config.embedding_size = args.embedding_size
    config.train_times = args.train_times
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.entity_total = entity_total
    config.relation_total = relation_total
    config.tem_total = tem_total

    model = model.TTransEModel(config)
    if USE_CUDA:
        model.cuda()
    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
    margin = autograd.Variable(floatTensor([config.margin]))

    # Training
    print(f'    🎯 Start Training ... ')
    trainBatchList = getBatchList(trainList, config.batch_size)
    print("✅ Batching completed.")
    print("Total number of batches:", len(trainBatchList))
    print("First batch size:", len(trainBatchList[0]) if trainBatchList else "No batches")
    for epoch in range(config.train_times):
        model.train()
        total_loss = floatTensor([0.0])
        random.shuffle(trainBatchList)
        start = time.time()
        for i, batchList in enumerate(trainBatchList):
            batch_start = time.time()

            pos_h_batch, pos_t_batch, pos_r_batch, pos_time_batch, neg_h_batch, neg_t_batch, neg_r_batch, neg_time_batch = getBatch_raw_all(batchList, config.entity_total)

            pos_h_batch = autograd.Variable(longTensor(pos_h_batch))
            pos_t_batch = autograd.Variable(longTensor(pos_t_batch))
            pos_r_batch = autograd.Variable(longTensor(pos_r_batch))
            pos_time_batch = autograd.Variable(longTensor(pos_time_batch))
            neg_h_batch = autograd.Variable(longTensor(neg_h_batch))
            neg_t_batch = autograd.Variable(longTensor(neg_t_batch))
            neg_r_batch = autograd.Variable(longTensor(neg_r_batch))
            neg_time_batch = autograd.Variable(longTensor(neg_time_batch))

            model.zero_grad()
            pos, neg = model(pos_h_batch, pos_t_batch, pos_r_batch, pos_time_batch, neg_h_batch, neg_t_batch, neg_r_batch, neg_time_batch)

            losses = config.loss_function(pos, neg, margin)
            #losses.backward()
            #optimizer.step()
            #total_loss += losses.data

            ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
            rel_embeddings = model.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
            tem_embeddings = model.tem_embeddings(torch.cat([pos_time_batch, neg_time_batch]))
            losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings) + loss.normLoss(tem_embeddings)
            losses.backward()
            optimizer.step()
            total_loss += losses.data

            if i % 100 == 0:
                print(f"🌀 Epoch {epoch}, Batch {i}/{len(trainBatchList)}, Loss: {losses.item():.4f}, Time per batch: {(time.time() - batch_start):.2f}s", flush=True)

        print(f"✅ Epoch {epoch} finished. Total Loss: {total_loss.item():.4f}, Time: {(time.time() - start):.2f}s", flush=True)

    model.eval()
    # --- Extract and Save Embeddings for Target Entities ---
    print(f"📂 Loading target entities from: {args.target_entity_file}")
    target_entities = load_targets(args.target_entity_file)

    print(f"📁 Loading entity2id mapping from: {os.path.join(data_path, 'entity2id.txt')}")
    entity2id_path = args.entity

    print(f"🧠 Extracting embeddings from model")
    entity_embeddings = model.ent_embeddings.weight.data.cpu().numpy()

    print(f"💾 Saving embeddings to: {args.output_path}")
    extract_embeddings_for_specific_entities(
        entity_list=target_entities,
        entity2id_path=entity2id_path,
        ent_embeddings=entity_embeddings,
        outPath=args.output_path,
        experimentName=args.experiment
    )