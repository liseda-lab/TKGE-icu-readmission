## 🎯 Goal
_This folder aims to generate the files needed for TKGS. Three types of files are gererated: Patient (P), Ontology (O), Merged. These files will be used to generate the traing set for the different embeddings. Here is a summary of the pairing based on the different KGS._

| KG Type | Files Requeired 
|-----------|-----------|
| _simple_ | Patient Files | 
| _semantic_ | Patient + Ontology Files (Merged)| 
| _temporal_ | Patient Files|
| _semantic-temporal_| Patient + Ontology Files (Merged)|

_These files will that be used to train the different embedding strategies._

## **❗️❕ Data availability ❗️❕**

To facilitate reproducibility, this step can be completlly skipped. The following download will provide a folder with just the files required for the next steps of our implementation. Please place this folder within the main folder of the project

````python
    download Url
````

## **📊 Usage**

1. Usign the [clinical-temporal-kg](https://github.com/liseda-lab/clinical-temporal-kg.git) repo, make the annotation files for ICU containing the patient's facts

2. Make the patient files - based on the annotations previously generated -  Please change the paths inside the python script - The name of the outFiles can also bechanged.
````python
    python3 makePatientsFiles.py
````

3. Make the ontology files - Please change the paths inside the python script
````python
    python3 makeOntologyFiles.py
````

4. Merge the files and re-atribute ids - Please change the paths inside the python script
````python
    python3 mergeFiles.py
    python3 mergeSummary.py
````

5. Create a data folder and place the following files inside individual folders (The file name may need to change based on the embedding specific requirements):

| semantic_notime | semantic_time | simple_notime | simple_time 
|-----------|-----------|-----------|-----------|
| MergedNodes.txt| MergedNodes.txt | PN.txt| PN.txt|
| MergedTriplesNoTime.txt | MergedTriplesRef.txt | PENT.txt| PREF.txt|
| MergedRelations.txt | MergedRelations.txt | PR.txt| PR.txt|
| MergedTrainNoTime.txt| MergedtrainRef.txt | PT.txt| PREFT.txt|

There are the files used to train the embeddings.