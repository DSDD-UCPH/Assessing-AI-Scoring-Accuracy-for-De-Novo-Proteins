python ./compare_monomer_multimer_binder_rmsd.py \
  --monomer-parquet ./monomer_binder_prediction_records.parquet \
  --multimer-rank1-parquet ./rank1_structures.parquet \
  --out-prefix monomer_vs_multimer