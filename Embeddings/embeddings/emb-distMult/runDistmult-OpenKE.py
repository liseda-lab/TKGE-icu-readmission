import sys
import os

#RUNTIME =
sys.path.append('Embeddings/embeddings/resources/OpenKE')

from openke.config import Trainer, Tester
from openke.module.model import ComplEx
from openke.module.loss import SoftplusLoss
from openke.module.loss import MarginLoss
from openke.module.loss.SoftplusLoss import SoftplusRankingLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader
import numpy as np


def extract_embeddings_for_specific_entities(entity_list, 
                                             entity2id_path, 
                                             ent_embeddings, 
                                             outPath, 
                                             experimentName
                                             ):
    """ 
    OutputPath = /home/folder
    """
    # Load entity2id mapping
    entity2id = {}
    with open(entity2id_path, 'r') as f:
        for line in f.readlines()[1:]:  # Skip the first line if it contains metadata
            try:
                entity, eid = line.strip().split('\t')
                entity2id[entity] = int(eid)
            except ValueError:
                continue

    # Open file to save embeddings with entity names
    with open(f'{outPath}{experimentName}_embeddings.txt', 'w') as f:
        for entity in entity_list:
            if entity in entity2id:
                eid = entity2id[entity]
                embedding = "\t".join(map(str, ent_embeddings[eid]))
                f.write(f"{entity}\t{embedding}\n")

    print(f"Saved embeddings for {len(entity_list)} entities (with matches) to {outPath}")


def main(dataPath, 
         outPath, 
         targtesFile, 
         experimentName
         ):
    """
    dataPath = /home/folder/  -  outPath = /home/folder  - targetFiles = /home/folder/taget.txt
    """
    # Define the expected file paths in the output directory
    output_files = [
        f'{dataPath}entity2id.txt',
        f'{dataPath}relation2id.txt',
        f'{dataPath}train2id.txt'
    ]

    # Check if all expected files exist
    assert all(os.path.exists(file) for file in output_files), (
        "Required output files are missing. Please ensure the following files are in the data directory:\n" +
        "\n".join(output_files)
    )

    # Load training data
    train_dataloader = TrainDataLoader(
        in_path = dataPath,
        nbatches = 512,
        threads = 8,
        sampling_mode = "normal",
        bern_flag = 1,
        filter_flag = 1,
        neg_ent = 10,
        neg_rel = 0
    )

    # set up complEx model
    complEx = ComplEx(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = 300
    )

    # define the loss function
    model = NegativeSampling(
	    model = complEx, 
	    #loss = SoftplusLoss(),
        loss = MarginLoss(margin = 5.0),
	    batch_size = train_dataloader.get_batch_size(), 
	    regul_rate = 1.0
    )

    # Train the model
    trainer = Trainer(model = model,
                    data_loader = train_dataloader,
                    train_times = 750,
                    alpha = 1e-3,
                    use_gpu = True,
                    opt_method="adam"
    )
    
    trainer.run()

    #Checkpoint
    complEx.save_checkpoint(f'{outPath}complEx.ckpt')
    complEx.load_checkpoint(f'{outPath}complEx.ckpt')
    
    # Get entity embeddings
    ent_embeddings = complEx.ent_re_embeddings.weight.cpu().data.numpy()
    with open(targtesFile, 'r') as f:
        entity_list = [line.strip() for line in f]
    
    print('Extracting The Embeddings')
    extract_embeddings_for_specific_entities(entity_list,f'{dataPath}entity2id.txt', ent_embeddings, outPath, experimentName)


if __name__ == '__main__':
    
    print('Script Starting ...')
    script, dataPath, outPath, targetFiles, experimentName = sys.argv
    main(dataPath, outPath, targetFiles, experimentName)