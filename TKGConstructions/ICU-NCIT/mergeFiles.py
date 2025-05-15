import os
from tabulate import tabulate
from tqdm import tqdm

def merge_files (files_path,start_id = 0):
    '''Merges entities from multiple files and reassigns new IDs starting from `start_id`.
    The only requirement is that the files have the same format (entity ID, entity name).'''
    nodes = {}
    count = 0
    repeats = set()
    for filepath in tqdm(files_path, desc="Merging files"):
        with open(filepath, "r") as f:
            lines = f.readlines()[1:]
            print(f'File {filepath} has: {len(lines)} unique entities')
            count += len(lines)
            for line in lines:
                entity, id = line.strip().split("\t")
                if entity not in nodes:
                    nodes[entity] = start_id
                    start_id += 1
                else:
                    repeats.add(entity)

    return nodes, count, repeats

def new_edges_notime(files_path, nodes, relations):
    '''Merges edges from multiple files and replaces entities and relations with new IDs.'''
    edges = []
    triples = []
    count = 0
    miss = 0
    for filepath in tqdm(files_path, desc="Edge Files"):
        with open(filepath, "r") as f:
            lines = f.readlines()[1:]
            print(f'File {filepath} has: {len(lines)} unique triples')
            count += len(lines)

            for line in lines:
                node1, node2, relation = line.strip().split("\t")
                try:
                    edges.append((nodes[node1], nodes[node2], relations[relation]))
                    triples.append((node1, node2, relation))
                except KeyError:
                    miss += 1

    return edges, count, miss, triples

def new_edges_time(files_path, nodes, relations):
    '''Merges edges from multiple files and replaces entities and relations with new IDs.'''
    edges = []
    triples = []
    count = 0
    miss = 0
    for filepath in tqdm(files_path, desc="Temporal Edge Files"):
        with open(filepath, "r") as f:
            lines = f.readlines()[1:]
            print(f'File {filepath} has: {len(lines)} unique facts')
            count += len(lines)

            for line in lines:
                if 'purl.obolibrary.org/obo/hasPrescriptions' in line.strip().split("\t")[2]:
                    node1, node2, relation, startTime, endTime = line.strip().split("\t")
                    try:
                        edges.append((nodes[node1], nodes[node2], relations[relation], startTime, endTime))
                        triples.append((node1, node2, relation, startTime, endTime))
                    except KeyError:
                        miss += 1
                else:
                    node1, node2, relation, time = line.strip().split("\t")
                    try:
                        edges.append((nodes[node1], nodes[node2], relations[relation], time))
                        triples.append((node1, node2, relation, time))
                    except KeyError:
                        miss += 1

    return edges, count, miss, triples

def new_edges_reference_time(files_path, nodes, relations):
    '''Merges edges from multiple files and replaces entities and relations with new IDs.'''
    edges = []
    triples = []
    count = 0
    miss = 0
    for filepath in tqdm(files_path, desc="Temporal Edge Files"):
        with open(filepath, "r") as f:
            lines = f.readlines()[1:]
            print(f'File {filepath} has: {len(lines)} unique facts')
            count += len(lines)

            for line in lines:
                node1, node2, relation, time = line.strip().split("\t")
                try:
                    edges.append((nodes[node1], nodes[node2], relations[relation], time))
                    triples.append((node1, node2, relation, time))
                except KeyError:
                    miss += 1

    return edges, count, miss, triples

def write_file(filename, data):
    with open(filename, "w") as f:
        f.write(f"{len(data)}\n")  # First line: number of entries  
        for line in data:
            f.write("\t".join(map(str, line)) + "\n")

print("Merging with ID reassignment Starting! 🚀")

nodes, countN, repeatN = merge_files(['TkgConstructions/ICU-NCIT/ON.txt', 'TkgConstructions/ICU-NCIT/PN.txt'])
relations, countR, repeatR = merge_files(['TkgConstructions/ICU-NCIT/OR.txt', 'TkgConstructions/ICU-NCIT/PR.txt'])
edges, count, miss, triples = new_edges_notime(['TkgConstructions/ICU-NCIT/OENT.txt', 'TkgConstructions/ICU-NCIT/PENT.txt'], nodes, relations)
edgesT, countT, missT, triplesT = new_edges_time(['TkgConstructions/ICU-NCIT/OE.txt', 'TkgConstructions/ICU-NCIT/PE.txt'], nodes, relations)
refEdges, countRef, missRef, refTriples = new_edges_reference_time(['TkgConstructions/ICU-NCIT/OE.txt', 'TkgConstructions/ICU-NCIT/PREF.txt'], nodes, relations)


print("Writing Merger Files! 📄📄📄")

write_file('TkgConstructions/ICU-NCIT/MergedNodes.txt', [(e, nodes[e]) for e in sorted(nodes.keys())])
write_file('TkgConstructions/ICU-NCIT/MergedRelations.txt', [(r, relations[r]) for r in sorted(relations.keys())])
write_file('TkgConstructions/ICU-NCIT/MergedTrainNoTime.txt', edges)
write_file('TkgConstructions/ICU-NCIT/MergedTrain.txt', edgesT)
write_file('TkgConstructions/ICU-NCIT/MergedTrainRef.txt', refEdges)
write_file('TkgConstructions/ICU-NCIT/MergedTriplesNoTime.txt', triples)
write_file('TkgConstructions/ICU-NCIT/MergedTriples.txt', triplesT)
write_file('TkgConstructions/ICU-NCIT/MergedTriplesRef.txt', refTriples)

print("Calculating and Writing Summary! 🧮 🧮")

data = [
    ("File Nodes 🎯", countN),
    ("Merger Nodes 🎯", len(nodes)),
    ("Union Nodes 🎯", len(repeatN)),
    ("Possible Unions in annotations", 5017), #Values need to be updated if changed
    ("File Relations", countR),
    ("Merger Relations", len(relations)),
    ("File Edges", count),
    ("Merger Edges", len(edges)),
    ("Missing Edges 🎯", miss),
    ("File Edges Time", countT),
    ("Merger Edges Time", len(edgesT)),
    ("Missing Relations Time 🎯", missT),
    ("Merger Triples", len(triples)),
    ("Merger Triples Time", len(triplesT)),
    ("Merger Edges Reference Time", len(refEdges)),
    ("Merger Triples Reference Time", len(refTriples))
    ]

table = tabulate(data, headers=["Item", "Count"], tablefmt="fancy_grid")

with open("TkgConstructions/ICU-NCIT/MergedSummary.txt", "w") as f:
    f.write("\n" + table)
print(table)
