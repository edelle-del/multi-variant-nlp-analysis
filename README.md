# Multi-Variant Text Analysis and Generation (BERT, GPT, and Text-GANs)

A comparative implementation of three distinct NLP model paradigms — **representation** (BERT), **generation** (GPT-2), and **adversarial generation** (a 1D-CNN Text-GAN) — fine-tuned and evaluated on a single, consistently-split corpus.

**Members:**
1. Lumabi, Edelle Gibben
2. Manuel, Craig Zyrus
3. Santillan, Daniel

---

## Project Objectives

1. Implement three distinct NLP model variants using pre-trained and custom deep learning architectures: BERT (Representation), GPT (Generation), and a 1D-CNN Sequence GAN (Adversarial Generation).
2. Establish a unified, automated data pipeline using Google Drive cloud synchronization to handle checkpoint saves and automated model reload logic.
3. Quantify performance limits, operating trade-offs, and tokenization impacts using a dual evaluation methodology (Discriminative vs. Generative metrics).

## Dataset

**Cornell Movie Review Data (Rotten Tomatoes)** via Hugging Face Datasets — [`cornell-movie-review-data/rotten_tomatoes`](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)

- 10,662 balanced movie review sentences, binary sentiment labels (0 = negative, 1 = positive)
- Splits: 8,530 train / 1,066 validation / 1,066 test
- The **same train/validation/test split** is reused across all three model variants
- Original source: Pang, B., & Lee, L. (2005). *Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales.* ACL 2005.

## Notebook Roadmap

| Part | Section | Output |
|---|---|---|
| 0 | Environment & Drive Setup | Installed libraries, mounted storage |
| 1 | Dataset Acquisition | `rotten_tomatoes` train/val/test splits |
| 2 | Variant 1 — BERT | Fine-tuned classifier + Precision/Recall/F1 |
| 3 | Variant 2 — GPT-2 | Fine-tuned generator + Perplexity + BLEU/ROUGE |
| 4 | Variant 3 — Text-GAN | Adversarially trained generator/discriminator + Accuracy/Precision/Recall/F1 + BLEU/ROUGE |
| 5 | Inference Verification | Reloaded models, qualitative generation samples |
| 6 | Comparison Matrix | Consolidated metrics table across all three variants |
| 7 | Analytical Discussion | Tokenization differences & GAN vs. autoregressive tradeoffs |

## Results Summary

*(From a genuine from-scratch training run — all cached checkpoints in Drive were cleared first, so these reflect real per-epoch training, not a cache reload. Full per-epoch tables are in `artifacts/TRAINING_HISTORY.md`.)*

| Model Variant | Primary Metric (Precision / Recall / F1) | Generative Quality Metric (BLEU / ROUGE / Perplexity) | Training Time/Epoch | Key Observations |
|---|---|---|---|---|
| **BERT** (`bert-base-uncased`) | P=0.8709 / R=0.8480 / F1=0.8593 | N/A (classification task) | 180.76s | Bidirectional WordPiece attention gives strong, stable classification performance; no native generation capability. |
| **GPT-2** (`distilgpt2`) | N/A (generative task) | BLEU≈0.0165, ROUGE-1≈0.0976, ROUGE-L≈0.0851, Perplexity≈56.90 | 83.35s | Autoregressive BPE decoding produces fluent, grammatical continuations even when lexical overlap with one reference is low. |
| **Text-GAN** (1D-CNN, word-level) | Acc=0.9961 / P=1.0000 / R=0.9922 / F1=0.9961 (discriminator) | BLEU≈0.0012, ROUGE-1≈0.0051, ROUGE-L≈0.0051 | 2.37s | Discriminator pulled ahead of the generator for most of training (oscillating, not flat) — see Part 7 of the notebook for the full epoch-by-epoch diagnosis. |

*Full discussion of why these numbers turned out this way — including the epoch-by-epoch discriminator/generator imbalance diagnosis for the GAN — is in **Part 7: Analytical Discussion** of the notebook.*

## Repository Structure

```
.
├── README.md
├── ACT_4_Multi_Variant_Text_Analysis_and_Generation__BERT__GPT__and_GANs_.ipynb
├── ACT_4_Lumabi_Manuel_Santillan_ipynb_-_Colab.pdf   # exported, fully-run notebook
└── artifacts/
    ├── TRAINING_HISTORY.md            # final per-model metrics + notes on epoch-level logging
    ├── training_history_summary.csv   # same data in machine-readable form
    └── TEST_INFERENCE_LOG.md          # standalone log of qualitative generation samples (GPT, Text-GAN) and BERT's classification metrics
```

## Model Checkpoints

Trained model weights are **not** committed to this repository (BERT/GPT checkpoints are several hundred MB each, and Git is not suited to storing binary model weights). All checkpoints are hosted on Google Drive instead:

**📁 [Model checkpoints — Google Drive](https://drive.google.com/drive/folders/1Ei7TVW4xTdU_at7BUAVdK5nYGUcwPBYt?usp=drive_link)**

The folder contains:
- `bert_results/` — fine-tuned `bert-base-uncased` classifier weights
- `gpt_results/` — fine-tuned `distilgpt2` generator weights
- `gan_generator_v2.pt` / `gan_discriminator_v2.pt` — trained Text-GAN generator/discriminator weights

The notebook's checkpoint logic automatically detects and loads these files from Drive if present, and trains from scratch otherwise — no manual setup is needed beyond mounting the Drive folder named `NLP_Assignment_Models` (the notebook creates this automatically if it doesn't exist).

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run all cells top-to-bottom (**Runtime → Run all**). A GPU runtime is recommended (**Runtime → Change runtime type → T4 GPU**) for the BERT and GPT-2 training cells.
3. When prompted, authorize Google Drive access — this is required for the checkpoint save/reload pipeline.
4. All metrics (Precision/Recall/F1, Perplexity, BLEU/ROUGE, Discriminator Accuracy) print inline and are consolidated automatically into the Part 6 comparison matrix.

## Training Histories & Test Inferences

Per-epoch training logs (loss per epoch, training time per epoch) print directly in the notebook's training cells whenever a model trains from scratch — the notebook's checkpoint-detection logic skips training and goes straight to evaluation if cached weights already exist in the linked Drive folder, which is what happened in the run captured in the submitted PDF. See [`artifacts/TRAINING_HISTORY.md`](artifacts/TRAINING_HISTORY.md) for the final metrics from that run plus notes on regenerating fresh per-epoch logs.

Qualitative test inference samples (GPT-2 prompt completions, Text-GAN decoded sentences, and BERT's classification metrics) are extracted as a standalone log in [`artifacts/TEST_INFERENCE_LOG.md`](artifacts/TEST_INFERENCE_LOG.md), in addition to appearing inline in the notebook's Part 3 and Part 5 cells.

## Dataset Repository Link

[`cornell-movie-review-data/rotten_tomatoes`](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) — Hugging Face Datasets Hub

## Analytical Discussion

A full write-up covering:
- How tokenization differences (WordPiece vs. BPE vs. custom word-level vocabulary) affected each model
- Why Precision/Recall/F1 and BLEU/ROUGE/Perplexity measure fundamentally different things, and why each variant needs a different metric set
- Why GANs struggle to train on discrete text sequences relative to autoregressive models like GPT — including a diagnosis of the discriminator-dominance failure mode observed in this run

...is included in **Part 7** of the notebook (also reproduced in the exported PDF).

## References

See the consolidated **References** section at the end of the notebook for full citations (BERT, GPT-2, GANs, SeqGAN, BLEU, ROUGE, and the dataset paper), and inline citations throughout the code cells.

## License

This project is submitted as coursework. Code is provided as-is for educational purposes.