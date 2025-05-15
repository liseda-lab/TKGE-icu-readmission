import os
import sys
import numpy as np
import datetime
import argparse

# ==== CONFIGURATION ====
start_time_str = "2018-01-01"
start_date = datetime.datetime.strptime(start_time_str, "%Y-%m-%d")
tem_dict = {
    '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
    '01m': 10, '02m': 11, '03m': 12, '04m': 13, '05m': 14, '06m': 15,
    '07m': 16, '08m': 17, '09m': 18, '10m': 19, '11m': 20, '12m': 21,
    '0d': 22, '1d': 23, '2d': 24, '3d': 25, '4d': 26, '5d': 27,
    '6d': 28, '7d': 29, '8d': 30, '9d': 31
}

# ==== INPUT AND OUTPUT PATHS ====
parser = argparse.ArgumentParser(description="Convert time column to indexed time IDs.")
parser.add_argument('--dataset', '-d', type=str, required=True, help='Name of the dataset folder')
parser.add_argument('--datapath', '-p', type=str, required=True, help='Path to the dataset directory')

args = parser.parse_args()
dataset = args.dataset
input_path = args.datapath
output_path = os.path.join(args.datapath, f"{dataset}")

#dataset = sys.argv[1]  # Provide your dataset name, e.g., 'myKG'
#input_path = f'../data/{dataset}/'
#output_path = f'data/{dataset}_TA/'
os.makedirs(output_path, exist_ok=True)

def generate_stat_file(triples_path, output_stat_path):
    entity_set = set()
    relation_set = set()

    for split in ['train2id', 'valid2id', 'test2id']:
        with open(os.path.join(triples_path, split + ".txt"), "r") as f:
            for line in f:
                h, r, t, *_ = line.strip().split("\t")
                entity_set.add(h)
                entity_set.add(t)
                relation_set.add(r)

    entity_total = len(entity_set)
    relation_total = len(relation_set)

    with open(output_stat_path, "w") as f:
        f.write(f"{entity_total} {relation_total} 0\n")

    print(f"Generated stat.txt with {entity_total} entities, {relation_total} relations.")

def convert_and_save(part):
    triples_file = os.path.join(input_path, part + "2id.txt")
    temporal_output_file = os.path.join(output_path, part + "_tem.npy")
    # Delete if it already exists
    if os.path.exists(temporal_output_file):
        os.remove(temporal_output_file)

    updated_triples_file = os.path.join(output_path, part + "2id.txt")

    temporal_features = []
    with open(triples_file, "r") as fr, open(updated_triples_file, "w") as fw:
        print()
        for line in fr:
            h, r, t, rel_time = line.strip().split("\t")
            rel_time = int(rel_time)  # assuming integer time offset

            # Convert relative time to calendar date
            abs_date = start_date + datetime.timedelta(hours=rel_time)
            date_str = abs_date.strftime("%Y-%m-%d")

            # Write updated triple with structured time string
            fw.write(f"{h}\t{r}\t{t}\t{date_str}\t0\n")

            # Temporal tokenization
            y, m, d = date_str.split("-")
            tem_ids = [tem_dict[ch + 'y'] for ch in y]  # year as 4 digits
            tem_ids.append(tem_dict[m + 'm'])           # month as 2 digits
            tem_ids += [tem_dict[ch + 'd'] for ch in d] # day as 2 digits
            temporal_features.append(tem_ids)

    np.save(temporal_output_file, np.array(temporal_features))

# ==== PROCESS TRAIN/VALID/TEST ====
for split in ['train', 'valid', 'test']:
    convert_and_save(split)

generate_stat_file(output_path, os.path.join(output_path, "stat.txt"))

print(f"Preprocessing complete for dataset '{dataset}'. Files saved to: {output_path}")