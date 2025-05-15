import os
import numpy as np
import argparse

def load_mappings(file_path):
    mappings = {}
    with open(file_path, "r") as f:
        content = f.readlines()[1:]
        total = len(content)
        for line in content:
            name, idx = line.strip().split("\t")
            mappings[name] = int(idx)
    print(f"Loaded {len(mappings)} mappings from {total}")
    return mappings

def convert_triples(file_path):
    """Converts triples to numpy format and prepares train2id.txt."""
    examples = []
    with open(file_path, "r") as f:
        content = f.readlines()[1:]
        for line in content:
            parts = line.strip().split("\t")
            if len(parts) == 4:  # (lhs, rel, rhs, timestamp)
                lhs, rel, rhs, timestamp = map(int, parts)
                examples.append([lhs, rhs, rel, timestamp])  # train2id format: lhs rhs rel time
    
    print(f"Converted {len(examples)} triples out of {len(content)}")
    for item in examples[4000000:4000010]:
        print(f'{item}\n')
    return examples, np.array([ex[-1] for ex in examples], dtype='int64')

def write_train2id(examples, output_path):
    with open(output_path, "w") as f:
        #f.write(f"{len(examples)}\n")   #FOR THIS EMBEDDING THE LENGTH is unecessary
        for ex in examples:
            f.write(f"{ex[0]}\t{ex[1]}\t{ex[2]}\t{ex[3]}\n")
    print(f"✅ Wrote train2id.txt with {len(examples)} examples.")

def write_stat_file(entity_file, relation_file, time_array, output_path):
    num_entities = sum(1 for _ in open(entity_file)) - 1
    num_relations = sum(1 for _ in open(relation_file)) - 1
    num_timestamps = len(set(time_array.tolist()))

    with open(os.path.join(output_path, "stat.txt"), "w") as f:
        f.write(f"{num_entities}\t{num_relations}\t{num_timestamps}")
    print(f"✅ Wrote stat.txt with {num_entities} entities, {num_relations} relations, and {num_timestamps} timestamps.")

def write_empty_files(data_dir,dataset):
    for split in ["valid2id.txt", "test2id.txt"]:
        open(os.path.join(data_dir, split), "w").close()
        print(f"✅ Created empty {split}")

    for tem in ["valid_tem.npy", "test_tem.npy"]:
        np.save(os.path.join(f'{data_dir}/{dataset}', tem), np.array([]))
        print(f"✅ Created empty {tem}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for TA_TransE")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--relation", type=str, required=True)
    parser.add_argument("--triples", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    load_mappings(args.entity)  # Just to confirm files are okay
    load_mappings(args.relation)

    examples, time_array = convert_triples(args.triples)
    write_train2id(examples, os.path.join(args.outdir, "train2id.txt"))
    new_path=f'{args.outdir}/{args.dataset}'
    os.makedirs(new_path, exist_ok=True)

    np.save(os.path.join(new_path, "train_tem.npy"), time_array)
    print("✅ Saved train_tem.npy")

    write_stat_file(args.entity, args.relation, time_array, args.outdir)
    write_empty_files(args.outdir, args.dataset)

    print("🏁 All set for TA_TransE training!")

    ########## CHECKS FOR OUR DATA SET ############
    max_relation_id = 4  # because you have 5 relations (0-indexed)
    with open(f"{args.outdir}/train2id.txt", "r") as f:
        lines = f.readlines()

    fmax_relation_id = 4  # you have 5 relations, IDs 0 to 4
    for i, line in enumerate(lines[1:], start=2):  # skip first line (count from 2)
        parts = line.strip().split()
        if len(parts) != 4:
            print(f"❌ Line {i} does not have 4 elements: {line.strip()}")
            continue
        h, r, t, ts = map(int, parts)
        if r < 0 or r > max_relation_id:
            print(f"❌ Invalid relation ID {r} at line {i}")