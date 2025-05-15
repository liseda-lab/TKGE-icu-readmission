import os
import pickle
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

def convert_triples(file_path, entities, relations):
    """Converts triples to numpy format."""
    examples = []
    with open(file_path, "r") as f:
        content = f.readlines()[1:]
        for line in content:
            parts = line.strip().split("\t")
            if len(parts) == 4:  # (entity_id, entity_id, relation_id, timestamp)
                lhs, rhs, rel, timestamp = parts
                examples.append([
                    int(lhs), int(rel), int(rhs), int(timestamp)  # Convert to integers
                ])
    print(f"Converted {len(examples)} triples out of {len(content)}")
    for item in examples[4000000:4000010]:
        print(f'{item}\n')
    return np.array(examples).astype('uint64')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert txt to pikle")
    parser.add_argument("--entity", type=str, required=True, help="Path to enetity file")
    parser.add_argument("--relation", type=str, required=True, help="Path to relations file")
    parser.add_argument("--triples", type=str, required=True, help="Path to triples file")
    args = parser.parse_args()
    #Process all data to a single dataSet
    entities = load_mappings(args.entity)
    realtions = load_mappings(args.relation)
    triples = convert_triples(args.triples, entities, realtions)
    with open("Embeddings/embeddings/resources/tkbc/tkbc/data/train.pickle", "wb") as f:
        pickle.dump(triples, f)
    empty_array = np.array([])  # Or np.zeros((0, 4))
    with open('Embeddings/embeddings/resources/tkbc/tkbc/data/test.pickle', 'wb') as f:
        pickle.dump(empty_array, f)
    with open('Embeddings/embeddings/resources/tkbc/tkbc/data/valid.pickle', 'wb') as f:
        pickle.dump(empty_array, f)
    with open('Embeddings/embeddings/resources/tkbc/tkbc/data/to_skip.pickle', 'wb') as f:
        pickle.dump(empty_array, f)

    print("✅ Data conversion complete! Only train.pickle generated.")
