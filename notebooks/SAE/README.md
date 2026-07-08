# SAE Notebook Guide

These notebooks reproduce the sparse-autoencoder (SAE), local-manifold, and LLM interpretation analyses for SLformer pair embeddings. 

## Configuration Files

- `SAE_training/config/train_config.yaml`
- `manifold/config/manifold_config.yaml`
- `LLM_pipeline/config/explainer_simulator_config.yaml`
- `doc/prompts/`

## Data Usage

The main run uses the aligned SLformer artifacts configured in `train_config.yaml`:

- Training SAE
    - `data/all_SL/mix9_alignment/  mix9_slformer_kg_crossemb.pkl`
    - `data/all_SL/mix9_alignment/  pred_mix9_slformer_kg_cv*.csv`
    - `data/all_SL/mix_slformer_kg_crossemb.pkl`, `data/all_SL/ pred_mix_slformer_kg_cv*.csv`, `data/merged_pred_true_slformer/  merged_pred_true_fold_*.csv`, and `data/saved_data/SL_train_test_data` for reconstruction validation.

- Intepretation
  - `data/IDH1-PRKDC-emb/IDH1_PRKDC_context_embeddings.npy` and `data/IDH1-PRKDC-emb/IDH1_PRKDC_context_meta.csv` for the special IDH1-PRKDC Glioma target.

>[!NOTE]
> The IDH1-PRKDC related embeddings are provided, for other data used, please visit the project data storage: https://doi.org/10.5281/zenodo.18733691

## Running Sequence

1. run `SAE_training/reconstruct_slformer.ipynb` to train the SLformer SAE and create `final.pt`, normalization arrays, and training metrics.
2. Run `SAE_training/SAE_analysis.ipynb` or `manifold/projection.ipynb` after the trained SAE exists to inspect latent activations and local score-direction projections.

3. For one target pair, run `LLM_pipeline/SAE_explainer_single.ipynb` first. It creates `feature_rank.csv`, `llm_interpretation_summary.csv`, and `interpretation_state.json` under the SAE run's `explanations/{cancer}_{primary}-{partner}/` folder.
   
4. Run `LLM_pipeline/LLM_output_single.ipynb` after the explainer notebook to build the final interpretation prompt and final interpretation files for the same target pair.
5. Run `LLM_pipeline/SAE_explainer_visualization.ipynb` after the explainer outputs exist to inspect the target-pair local geometry and atom visualization.

>[!NOTE]
> `notebooks/LLM_interpretation/prompt_api` is the wrapped API specifically for the task, we recommend no modification; Running `/notebooks/LLM_interpretation` requires configuring `notebooks/LLM_interpretation/prompt_api/client_config.yaml`.

