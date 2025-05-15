from tabulate import tabulate

def count_lines(filepath):
    """Returns the number of actual data lines in a file (excluding the first line)."""
    with open(filepath, "r") as f:
        return len(f.readlines()) - 1  # Ignore first line (header count)

# File paths
patient_nodes_file = "TkgConstructions/ICU-NCIT/PN.txt"
ontology_nodes_file = "TkgConstructions/ICU-NCIT/ON.txt"
merged_nodes_file = "TkgConstructions/ICU-NCIT/MergedNodes.txt"

patient_relations_file = "TkgConstructions/ICU-NCIT/PR.txt"
ontology_relations_file = "TkgConstructions/ICU-NCIT/OR.txt"
merged_relations_file = "TkgConstructions/ICU-NCIT/MergedRelations.txt"

patient_edges_file = "TkgConstructions/ICU-NCIT/PT.txt"
ontology_edges_file = "TkgConstructions/ICU-NCIT/OT.txt"
merged_edges_file = "TkgConstructions/ICU-NCIT/MergedTrain.txt"

patient_train_file = "TkgConstructions/ICU-NCIT/PTNT.txt"
ontology_train_file = "TkgConstructions/ICU-NCIT/OTNT.txt"
merged_train_file = "TkgConstructions/ICU-NCIT/MergedTrainNoTime.txt"

# Count lines
ontology_nodes_count = count_lines(ontology_nodes_file)
patient_nodes_count = count_lines(patient_nodes_file)
merged_nodes_count = count_lines(merged_nodes_file)

ontology_relations_count = count_lines(ontology_relations_file)
patient_relations_count = count_lines(patient_relations_file)
merged_relations_count = count_lines(merged_relations_file)

ontology_edges_count = count_lines(ontology_edges_file)
patient_edges_count = count_lines(patient_edges_file)
merged_edges_count = count_lines(merged_edges_file)

ontology_train_count = count_lines(ontology_train_file)
patient_train_count = count_lines(patient_train_file)
merged_train_count = count_lines(merged_train_file)

# Prepare data for tabulation
table_data = [
    ["Nodes", ontology_nodes_count, patient_nodes_count, ontology_nodes_count + patient_nodes_count, merged_nodes_count],
    ["Relations", ontology_relations_count, patient_relations_count, ontology_relations_count + patient_relations_count, merged_relations_count],
    ["Edges", ontology_edges_count, patient_edges_count, ontology_edges_count + patient_edges_count, merged_edges_count],
    ["Train", ontology_train_count, patient_train_count, ontology_train_count + patient_train_count, merged_train_count]
]

# Print table
headers = ["Category", "Ontology Count", "Patient Count", "Expected Merged", "Actual Merged"]
print(tabulate(table_data, headers=headers, tablefmt="pretty"))

# Save to file
output_file = "TkgConstructions/ICU-NCIT/MergingStats.txt"
with open(output_file, "w") as f:
    f.write(tabulate(table_data, headers=headers, tablefmt="pretty"))

print(f"Statistics saved to {output_file}")