import rdflib
import os
from tqdm import tqdm
from tabulate import tabulate

class OntologyProcessor:
    def __init__(self, ontology_files):
        self.ontology_files = ontology_files
        self.kg = rdflib.Graph()
        self.entities = set()
        self.relations = set()
        self.triples = []
        self.entity_to_id = {}
        self.start_entity_id = 500000
        self.start_relation_id = 1000
        self.relation_to_id = {}
        self.file_formats = {
            ".rdf": "xml",
            ".owl": "xml",
            ".ttl": "turtle",
        }

        # Relevant relations to extract
        self.target_relations = {
            rdflib.RDFS.subClassOf,
            rdflib.OWL.equivalentClass
            #rdflib.OWL.disjointWith
            #rdflib.RDFS.label
            #rdflib.URIRef("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#has_physiologic_effect")
        }

    def load_ontologies(self):
        """Loads all ontology files into the RDFLib graph."""
        for file in tqdm(self.ontology_files, desc="Loading Ontologies", unit="ontology"):
            ext = os.path.splitext(file)[-1].lower()
            format = self.file_formats.get(ext, None)
            if format:
                try:
                    self.kg.parse(file, format=format)
                except Exception as e:
                    print(f"Error parsing {file}: {e}")

    def process_triples(self):
        """Extracts only the relevant triples from the ontology."""
        for s, p, o in tqdm(self.kg, desc="Processing Triples", unit="triple"):
            if p in self.target_relations:
                self.entities.add(str(s.strip()))
                self.relations.add(str(p.strip()))
                self.entities.add(str(o.strip()))
                self.triples.append((str(s.strip()), str(p.strip()), str(o.strip())))

    def assign_ids(self):
        """Assigns unique incremental IDs to entities and relations."""
        self.entity_to_id = {entity: i for i, entity in enumerate(sorted(self.entities), start=self.start_entity_id)}
        self.relation_to_id = {relation: i for i, relation in enumerate(sorted(self.relations), start=self.start_relation_id)}
    
    def save_files(self, output_folder):
        """Writes entities, relations, and triples (with/without IDs) to text files."""
        os.makedirs(output_folder, exist_ok=True)

        def write_file(filename, data):
            with open(os.path.join(output_folder, filename), "w") as f:
                f.write(f"{len(data)}\n")  # First line: number of entries
                for line in data:
                    f.write("\t".join(map(str, line)) + "\n")
        
        missing_entities = set()
        missing_relations = set()
        triples_with_ids = []

        for s, p, o in self.triples:
            if s not in self.entity_to_id:
                missing_entities.add(s)
                continue
            if o not in self.entity_to_id:
                missing_entities.add(o)
                continue
            if p not in self.relation_to_id:
                missing_relations.add(p)
                continue

            triples_with_ids.append((self.entity_to_id[s], self.relation_to_id[p], self.entity_to_id[o]))

        write_file("ON.txt", [(e, self.entity_to_id[e]) for e in sorted(self.entities)])
        write_file("OR.txt", [(r, self.relation_to_id[r]) for r in sorted(self.relations)])
        write_file("OE.txt", [(s,o,p,0) for s,p,o in self.triples])
        write_file("OENT.txt", [(s,o,p) for s,p,o in self.triples])
        write_file("OT.txt", [(s,o,p,0) for s, p, o in triples_with_ids])
        write_file("OTNT.txt", [(s,o,p) for s, p, o in triples_with_ids])

        if missing_entities:
            print(f"Warning: {len(missing_entities)} missing entities. Example: {list(missing_entities)[:5]}")
        if missing_relations:
            print(f"Warning: {len(missing_relations)} missing relations. Example: {list(missing_relations)[:5]}")
        
        with open(os.path.join(output_folder,"Ontology_Summary.txt"), "w") as f:
                f.write(f"Warning: {len(missing_entities)} missing entities.\n")
                f.write(f"Warning: {len(missing_relations)} missing relations.")



    def run(self, output_folder):
        """Runs the full processing pipeline."""
        print("Processing ontologies...")
        self.load_ontologies()
        self.process_triples()

        print("Assigning IDs...")
        self.assign_ids()

        print("Saving files...")
        self.save_files(output_folder)
        print(f"Processing complete! Files saved in {output_folder}")

        data = [
            ("entities", len(self.entities)), 
            ("relations", len(self.relations)), 
            ("triples", len(self.triples))
            ]

        table = tabulate(data, headers=["Item", "Count"], tablefmt="fancy_grid")

        with open(os.path.join(output_folder,"Ontology_Summary.txt"), "a") as f:
            f.write("\n" + table)
    
        print(table)

if __name__ == "__main__":
    ontology_files = ["<path_to_your_NCIT_Ontology>"]  # Add your ontology files here
    os.makedirs(f"TkgConstructions/ICU-NCIT", exist_ok=True)
    output_folder = "TkgConstructions/ICU-NCIT"

    processor = OntologyProcessor(ontology_files)
    processor.run(output_folder)