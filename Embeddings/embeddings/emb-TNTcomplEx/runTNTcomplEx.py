import torch
import numpy as np
import sys
import os
import argparse
import random
#from typing import Dict
#import logging

sys.path.append("Embeddings/embeddings/resources/tkbc")

from tkbc import optimizers
from tkbc.optimizers import TKBCOptimizer, IKBCOptimizer
from tkbc.models import TNTComplEx, TComplEx
from tkbc.datasets import TemporalDataset
from tkbc.regularizers import N3, Lambda3
from torch import optim

def load_targets(target_file):
    with open(target_file, "r") as f:
        targets = [line.strip() for line in f]
    return set(targets)

def extract_embeddings_for_specific_entities(entity_list, entity2id_path, ent_embeddings, outPath, experimentName):
    entity2id = {}
    with open(entity2id_path, 'r') as f:
        for line in f.readlines()[1:]:  # Skip the first line if it contains metadata
            try:
                entity, eid = line.strip().split('\t')
                entity2id[entity] = int(eid)
            except ValueError:
                continue

    with open(f'{outPath}/{experimentName}_embeddings.txt', 'w') as f:
        for entity in entity_list:
            if entity in entity2id:
                eid = entity2id[entity]
                embedding = "\t".join(map(str, ent_embeddings[eid]))
                f.write(f"{entity}\t{embedding}\n")

    print(f"Saved embeddings for {len(entity_list)} entities (with matches) to {outPath}")

def train_and_extract_embeddings(dataset_path, model_name, target_file, entity2id_path, output_file, experiment_name, rank=300, max_epochs=25, batch_size=1024, learning_rate=1e-3, emb_reg=0., time_reg=0., no_time_emb=False):
    # Load dataset
    dataset = TemporalDataset(dataset_path)
    sizes = dataset.get_shape()
    print(f"Dataset loaded: {sizes}")

    # Initialize model
    model = {
        'TComplEx': TComplEx(sizes, rank, no_time_emb=no_time_emb),
        'TNTComplEx': TNTComplEx(sizes, rank, no_time_emb=no_time_emb),
    }[model_name]

    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    emb_reg = N3(emb_reg)
    time_reg = Lambda3(time_reg)
    optimizer = TKBCOptimizer(model, emb_reg, time_reg, opt, batch_size=batch_size)

    for epoch in range(max_epochs):
        examples = torch.from_numpy(dataset.get_train().astype('int64')).cuda()
        print(f'⏳ Training epoch:{epoch}')

        ############################################
        num_samples = min(5, examples.shape[0])  # Ensure we don't exceed the available number of samples
        random_indices = random.sample(range(examples.shape[0]), num_samples)
        random_examples = examples[random_indices]
        print("Random snapshot of training examples:")
        print(random_examples)
        ############################################
        
        model.train()
        optimizer.epoch(examples)

        print(f'    ✅ Epoch {epoch + 1}/{max_epochs} completed')

    # Extract embeddings
    target_entities = load_targets(target_file)    
    entity_embeddings = model.embeddings[0].weight.detach().cpu().numpy()
    extract_embeddings_for_specific_entities(target_entities, entity2id_path, entity_embeddings, output_file, experiment_name)

    print('         ✅✅✅ We made it ✅✅✅')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TComplex/TNTComplex and extract embeddings")
    parser.add_argument('--dataset', type=str, required=True, help="Path to train.pickle")
    parser.add_argument('--model', choices=['TComplEx', 'TNTComplEx'], required=True, help="Model type")
    parser.add_argument('--targets', type=str, required=True, help="Path to targets.txt")
    parser.add_argument('--entity2id', type=str, required=True, help="Path to entity2id file")
    parser.add_argument('--output', type=str, required=True, help="Path to save embeddings")
    parser.add_argument('--experiment', type=str, required=True, help="Experiment name for output file")
    parser.add_argument('--rank', type=int, default=300, help="Embedding rank/Size")
    parser.add_argument('--max_epochs', type=int, default=750, help="Training epochs") #25       #50
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size") #1024         #512
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate") #1.3 -> 7 #1e-2
    parser.add_argument('--emb_reg', type=float, default=0., help="Embedding regularization strength")
    parser.add_argument('--time_reg', type=float, default=0., help="Timestamp regularization strength")
    parser.add_argument('--no_time_emb', action='store_true', help="Disable time embeddings")
    
    args = parser.parse_args()
    
    train_and_extract_embeddings(
        dataset_path=args.dataset,
        model_name=args.model,
        target_file=args.targets,
        entity2id_path=args.entity2id,
        output_file=args.output,
        experiment_name=args.experiment,
        rank=args.rank,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        emb_reg=args.emb_reg,
        time_reg=args.time_reg,
        no_time_emb=args.no_time_emb
    )
