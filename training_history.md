# Training History

This folder's `training_history_summary.csv` captures the **final validation/test metrics** for each model variant from the run documented in the submitted PDF (`ACT_4_Lumabi_Manuel_Santillan_ipynb_-_Colab.pdf`).

## Why there's no per-epoch loss curve

All three models (BERT, GPT-2, Text-GAN) were reloaded from cached checkpoints in Google Drive during this run — the notebook's checkpoint-detection logic (`if os.path.exists(...)`) found existing trained weights and **skipped the training loop entirely**, going straight to evaluation. This is by design (it's what makes the notebook fast to re-run for grading/demo purposes), but it means no fresh per-epoch loss values were printed in this particular session.

Per-epoch loss curves for BERT and GPT *are* available in the Hugging Face `Trainer`'s internal logs (`trainer.state.log_history`) whenever the models are trained from scratch — e.g. on a fresh Colab runtime with an empty `SAVE_DIR`, or by deleting the cached checkpoints in the linked [Google Drive folder](https://drive.google.com/drive/folders/1Ei7TVW4xTdU_at7BUAVdK5nYGUcwPBYt?usp=drive_link) before running. The Text-GAN's training loop prints a `D loss` / `G loss` / `time` line every epoch when it trains from scratch (see the notebook's GAN training cell), which is the most direct way to capture its full per-epoch history.

## What's captured here instead

Final metrics at evaluation time, per model:

| Model | Key metric | Value |
|---|---|---|
| BERT | Validation F1 | 0.8692 |
| GPT-2 | Validation Perplexity | 56.85 |
| Text-GAN | Discriminator Accuracy | 1.0000 |

Full breakdown with precision/recall, BLEU/ROUGE, and notes is in `training_history_summary.csv`.
