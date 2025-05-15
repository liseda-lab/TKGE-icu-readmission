import os
import glob
import re
from tqdm import tqdm
import sys
from tabulate import tabulate

sys.path.append("TkgConstructions")
from timeHandlingModule import convert_to_fixed_intervals

class NodeManager:
    """Handles unique nodes and their assigned IDs."""
    def __init__(self):
        self.node_to_id = {}
        self.counter = 0  # IDs start from 1

    def get_id(self, node):
        """Returns the ID for a node, assigning a new one if necessary."""
        if node not in self.node_to_id:
            self.node_to_id[node] = self.counter
            self.counter += 1
        return self.node_to_id[node]

    def get_all_nodes(self):
        """Returns all nodes sorted by ID."""
        return sorted(self.node_to_id.items(), key=lambda x: x[1])
    
    def get_dided_nodes(self):
        """Returns all nodes sorted by ID."""
        return sorted(self.node_to_id.keys())
    

class RelationManager:
    """Handles unique relations and their assigned IDs."""
    def __init__(self):
        self.relation_to_id = {}
        self.counter = 0  # IDs start from 1

    def extract_relation(self, filename):
        """Extracts relation name from filename using regex (e.g., ICUAnnotPrescriptions.tsv -> Prescriptions)."""
        match = f"http://purl.obolibrary.org/obo/has{re.sub(r'^.*?ICUAnnot', '', filename)}"  # Removes everything up to and including 'ICUAnnot'
        return match.replace(".tsv", "").strip()

    def get_id(self, relation):
        """Returns the ID for a relation, assigning a new one if necessary."""
        if relation not in self.relation_to_id:
            self.relation_to_id[relation] = self.counter
            self.counter += 1
        return self.relation_to_id[relation]

    def get_all_relations(self):
        """Returns all relations sorted by ID."""
        return sorted(self.relation_to_id.items(), key=lambda x: x[1])
    

class AdmissionTimeManager:
    def __init__(self, admission_file):
        self.node_admission_time = {}
        self._load_admission_times(admission_file)

    def _load_admission_times(self, admission_file):
        with open(admission_file, "r") as f:
            for line in f:
                node, admission_time = line.strip().split("\t")
                self.node_admission_time[node] = admission_time

    def get_admission_time(self, node):
        return self.node_admission_time.get(node, None)  
    
    def has_admission_time(self, node):
        return node in self.node_admission_time
    
class TripleProcessor:
    """Processes TSV files to extract nodes, relations, and generate quadruples."""
    def __init__(self, input_folder, admission_file):
        self.input_folder = input_folder
        self.node_manager = NodeManager()
        self.relation_manager = RelationManager()
        self.admission_manager = AdmissionTimeManager(admission_file)
        self.edges = set()
        self.edges_no_time = set()
        self.train_edges = set()
        self.train_edges_no_time = set()
        self.reference_train = set()
        self.reference_edges = set()
        self.admission_issues = 0
        self.time_issues = 0
        self.nt_duplicate_count = 0

    def process_files(self):
        """Processes all TSV files in the input folder."""
        for file_path in tqdm(glob.glob(os.path.join(self.input_folder, "*.tsv")),  desc="Processing Files", unit="File"):
            relation = self.relation_manager.extract_relation(os.path.basename(file_path))
            relation_id = self.relation_manager.get_id(relation)

            with open(file_path, "r") as f:
                # 
                for line in f:
                    node1, node2, time = line.strip().split("\t")
                    node1 = f"http://purl.obolibrary.org/obo/{parts[0]}-{parts[1]}-{parts[2]}" if "http" not in (parts := node1.split(",")) else node1

                    node1_id = self.node_manager.get_id(node1)
                    node2_id = self.node_manager.get_id(node2)

                    # Exclude entries where the node has no admission time
                    if not self.admission_manager.has_admission_time(node1):
                        self.admission_issues += 1
                        continue

                    if 'Prescriptions' in relation:
                        begin_time, end_time = time.split(",")
                        # Exclude entries where time is invalid
                        if begin_time == "NULL" or begin_time == None or len(begin_time) == 0:
                            self.time_issues += 1
                            continue
                        
                        #Time
                        self.edges.add((node1, node2, relation, begin_time, end_time))
                        self.train_edges.add((node1_id, node2_id, relation_id, begin_time, end_time))

                        #ReferenceTime
                        admissionTime = self.admission_manager.get_admission_time(node1)
                        referenceTime = convert_to_fixed_intervals(begin_time, admissionTime)
                        if referenceTime < 0:
                            referenceTime = 0

                        self.reference_train.add((node1_id, node2_id, relation_id, referenceTime))
                        self.reference_edges.add((node1, node2, relation, referenceTime))

                        #No Time - Know how many of the quads are the same without time
                        if (node1, node2 ,relation) in self.edges_no_time:
                            self.nt_duplicate_count += 1

                        self.edges_no_time.add((node1, node2, relation))
                        self.train_edges_no_time.add((node1_id, node2_id, relation_id))
                        
                    else:
                        # Exclude entries where time is invalid
                        if time == "NULL" or time == None or len(time) == 0:
                            self.time_issues += 1
                            continue

                        #Time
                        self.edges.add((node1, node2, relation, time))
                        self.train_edges.add((node1_id, node2_id, relation_id, time))

                        #ReferenceTime
                        admissionTime = self.admission_manager.get_admission_time(node1)
                        referenceTime = convert_to_fixed_intervals(time, admissionTime)
                        if referenceTime < 0:
                            referenceTime = 0
                            
                        self.reference_train.add((node1_id, node2_id, relation_id, referenceTime))
                        self.reference_edges.add((node1, node2, relation, referenceTime))

                        #No Time - Know how many of the quads are the same without time
                        if (node1, node2, relation) in self.edges_no_time:
                            self.nt_duplicate_count += 1

                        self.edges_no_time.add((node1, node2, relation))
                        self.train_edges_no_time.add((node1_id, node2_id, relation_id))
                
    def get_data(self):
        """Returns processed data for writing."""
        return {
            "nodes": self.node_manager.get_all_nodes(),
            "dided_nodes": self.node_manager.get_dided_nodes(),
            "relations": self.relation_manager.get_all_relations(),
            "edges": self.edges,
            "train_edges": self.train_edges,
            "edges_no_time": self.edges_no_time,
            "train_edges_no_time": self.train_edges_no_time,
            "reference_edges": self.reference_edges,
            "reference_train": self.reference_train,
            "nt_duplicate_count": self.nt_duplicate_count,
            "admission_issues": self.admission_issues,
            "time_issues": self.time_issues
        }

class FileWriter:
    """Handles writing output files with required formatting."""
    @staticmethod
    def write_file(file_path, data, is_numeric=False):
        """Writes data to a file with the first line as the number of lines."""
        with open(file_path, "w") as f:
            f.write(f"{len(data)}\n")  # First line: Number of lines
            for item in data:
                f.write("\t".join(map(str, item)) + "\n")
    
    def write_extra_file(file_path, data, is_numeric=False):
        """Writes data to a file with the first line as the number of lines."""
        with open(file_path, "w") as f:
            for item in data:
                f.write(item + "\n")

def main(input_folder, output_folder,admission_file):
    os.makedirs(output_folder, exist_ok=True)

    print("Processing files...")
    processor = TripleProcessor(input_folder, admission_file)
    processor.process_files()
    data = processor.get_data()

    print("Writing output files...")
    # Write output files
    FileWriter.write_file(os.path.join(output_folder, "PN.txt"), data["nodes"])
    FileWriter.write_extra_file(os.path.join(output_folder, "Targets.txt"), data["dided_nodes"])
    FileWriter.write_file(os.path.join(output_folder, "PR.txt"), data["relations"])
    FileWriter.write_file(os.path.join(output_folder, "PE.txt"), data["edges"])
    FileWriter.write_file(os.path.join(output_folder, "PENT.txt"), data["edges_no_time"])
    FileWriter.write_file(os.path.join(output_folder, "PT.txt"), data["train_edges"])
    FileWriter.write_file(os.path.join(output_folder, "PTNT.txt"), data["train_edges_no_time"])
    FileWriter.write_file(os.path.join(output_folder, "PREF.txt"), data["reference_edges"])
    FileWriter.write_file(os.path.join(output_folder, "PREFT.txt"), data["reference_train"])

    print("Summary:")
    data = [
        ("Nodes", len(data["nodes"])),
        ("Relations", len(data["relations"])),
        ("Edges", len(data["edges"])),
        ("Train Edges", len(data["train_edges"])),
        ("Edges No Time", len(data["edges_no_time"])),
        ("Train Edges No Time", len(data["train_edges_no_time"])),
        ("Reference Edges", len(data["reference_edges"])),
        ("Reference Train Edges", len(data["reference_train"])),
        ("Non-Time Duplicate Count", data["nt_duplicate_count"]),
        ("Admission Issues", data["admission_issues"]),
        ("Time Issues", data["time_issues"])
    ]

    table = tabulate(
        data, 
        headers=["Statistic", "Amount"], 
        tablefmt="grid"
    )
    with open(os.path.join(output_folder,"Summary.txt"), "w") as f:
        f.write(table)
    
    print(table)

if __name__ == "__main__":
    input_folder = "DataSetConstruction/TemporalFacts/IcuStayFacts"   #Path to folder after construction
    output_folder = "TkgConstructions/ICU-NCIT"
    admission_file = "TkgConstructions/admissionTimes.tsv"  #Path to the admission times file

    main(input_folder, output_folder, admission_file)