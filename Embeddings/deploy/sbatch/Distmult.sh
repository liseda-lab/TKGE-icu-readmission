#!/bin/bash
#SBATCH --Something
#ADD YOUR SLURM OPTIONS HERE

#Structure of the script:
# 1. Script
# 2. Data Directory
# 3. Output Directory
# 4. Targetsfile Path
# 5. Name of the output file - model_icuNcit_KgType_dim


uv venv .venv
#SEMANTIC LAYER EMBEDDINGS
uv run python Embeddings/embeddings/emb-distMult/runDistmult-OpenKE.py \
      /datasets/ICU/Ncit_semantic_notime/ \
      /path_to_out_dir \
      /datasets/Targets.txt \
      distMult_icuNcit_semantic_300

#SIMPLE EMBEDDINGS
uv run python Embeddings/embeddings/emb-distMult/runDistmult-OpenKE.py \
      /datasets/ICU/Ncit_simple_notime/ \
      /path_to_out_dir \
      /datasets/Targets.txt \
      distMult_icuNcit_simple_300 