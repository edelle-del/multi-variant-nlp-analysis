# Training History

This folder's `training_history_summary.csv` captures the **full per-epoch training history** for all three model variants, from a genuine from-scratch run (all cached checkpoints in Google Drive were deleted before this run, forcing every model to train rather than reload).

## BERT (`bert-base-uncased`, 2 epochs)

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | — (no log) | 0.3344 | 0.8709 | 0.8480 | 0.8593 |
| 2 | 0.3276 | 0.3647 | 0.8462 | 0.8668 | 0.8563 |

Total training time: 361.5s → **180.76s / epoch**. Final validation metrics (best checkpoint): P=0.8709, R=0.8480, F1=0.8593.

## GPT-2 (`distilgpt2`, 2 epochs)

| Epoch | Training Loss | Validation Loss |
|---|---|---|
| 1 | 4.3682 | 4.0728 |
| 2 | 4.0293 | 4.0413 |

Total training time: 166.7s → **83.35s / epoch**. Final perplexity: **56.9005**.

## Text-GAN (1D-CNN, 15 epochs, trained fully from scratch — no pre-trained weights exist for this architecture)

| Epoch | D loss | G loss | Time (s) |
|---|---|---|---|
| 1 | 0.8697 | 0.7267 | 3.11 |
| 2 | 0.5506 | 1.1714 | 2.25 |
| 3 | 0.4673 | 1.4107 | 2.28 |
| 4 | 0.1860 | 2.6171 | 2.33 |
| 5 | 0.0849 | 3.3273 | 2.32 |
| 6 | 0.0380 | 4.4083 | 2.31 |
| 7 | 0.0424 | 4.6157 | 2.31 |
| 8 | 0.0532 | 4.7705 | 2.31 |
| 9 | 0.0750 | 4.4268 | 2.31 |
| 10 | 0.1195 | 4.1312 | 2.34 |
| 11 | 0.1038 | 4.0416 | 2.33 |
| 12 | 0.1505 | 4.3992 | 2.32 |
| 13 | 0.1007 | 5.5270 | 2.32 |
| 14 | 0.0474 | 6.9635 | 2.33 |
| 15 | 0.0928 | 8.1953 | 2.34 |

Average: **2.37s / epoch**. Final held-out evaluation: Discriminator Accuracy=0.9961, Precision=1.0000, Recall=0.9922, F1=0.9961; Generator BLEU=0.0012, ROUGE-1=0.0051, ROUGE-L=0.0051.

**What the curve shows:** discriminator loss drops sharply through epoch 6 while generator loss climbs (the discriminator pulling ahead), the generator partially recovers by epoch 11, then the discriminator pulls ahead again toward epoch 15. This oscillating-but-discriminator-favored pattern is discussed in the notebook's Part 7 analytical write-up as the direct cause of the GAN's low generative quality relative to GPT.

## Reproducing this from scratch

1. In the linked [Google Drive folder](https://drive.google.com/drive/folders/1Ei7TVW4xTdU_at7BUAVdK5nYGUcwPBYt?usp=drive_link), delete `bert_results/`, `gpt_results/`, `gan_generator_v2.pt`, and `gan_discriminator_v2.pt`.
2. Run the notebook top-to-bottom (Runtime → Run all) on a GPU runtime.
3. Each model's `if os.path.exists(...)` checkpoint check will fail (since the files are gone), so every variant trains from scratch and prints its full per-epoch table/loss log, exactly as captured above.