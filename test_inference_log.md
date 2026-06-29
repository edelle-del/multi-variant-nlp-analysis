# Test Inference Log

Qualitative inference outputs from all three model variants, captured from the from-scratch training run documented in the latest submitted PDF (`ACT_4_Lumabi_Manuel_Santillan_ipynb_-_Colab.pdf`). This is a standalone extract — the full code that produced these outputs lives in the notebook's Part 3 (GPT BLEU/ROUGE) and Part 5 (Inference Verification) cells.

---

## 1. BERT — Classification (quantitative, not generative)

BERT has no generation capability — its "inference" is a classification decision (positive/negative sentiment), not free text. Its qualitative output is therefore reported via metrics rather than sample text:

| Metric | Value |
|---|---|
| Precision | 0.8709 |
| Recall | 0.8480 |
| F1 | 0.8593 |
| Validation loss (final) | 0.3345 |

---

## 2. GPT-2 (`distilgpt2`) — Generative Inference

### 2a. Domain-specific prompt completion (Part 5)

**Prompt:** `"This movie was"`

| # | Generated continuation |
|---|---|
| 1 | "This movie was so much better than this one that I felt it deserved to be made for me... and I'll admit this is a little of a disa[ppointment]..." |
| 2 | "This movie was made with a hollywood style, and a few great performances. the film's script is an old-fashioned jock, and the acto[rs]..." |
| 3 | "This movie was designed to be a fun and entertaining exercise in the way it does the tedious work of a movie. the film is a bit of a..." |

Decoding strategy: `do_sample=True, top_k=50, top_p=0.95, temperature=0.7` (Fan et al. 2018 top-k + Holtzman et al. 2019 nucleus sampling).

### 2b. Prompt → ground-truth-continuation comparisons, with BLEU (Part 3)

Each row: a real test-set sentence split into a 4-token prompt + the actual continuation, compared against GPT's own continuation from that same prompt.

| Prompt | Reference continuation | Generated continuation | BLEU |
|---|---|---|---|
| "noyce creates a film" | "of near-hypnotic physical beauty even as he tells a story as horrifying as any in the heart-breakingly extensive ann..." | "that is as compelling as the one it is compelling as the one it is compelling. it's the kind of film that is as fascinating as the..." | 0.0146 |
| "[scherfig] has made a" | "movie that will leave you wondering about the characters' lives after the clever credits roll." | "considerable amount of effort to make the film a documentary about human rights and the impact of a long-standing conflict" | 0.0172 |
| "the dramatic scenes are" | "frequently unintentionally funny, and the action sequences -- clearly the main event -- are surprisingly uninvolving." | "all but complete, but the film still retains a sense of humor and charm. the characters are never dulled. the characters are oft[en]..." | 0.0159 |

**Aggregate (29 scored test samples):** Avg BLEU = 0.0165, Avg ROUGE-1 = 0.0976, Avg ROUGE-L = 0.0851.

*Note on low BLEU despite fluent output:* BLEU/ROUGE penalize any lexical divergence from one specific reference continuation. The generated text above is grammatical and topically on-point, but uses different words/phrasing than the one ground-truth continuation it's scored against — this is expected and is discussed further in the notebook's Part 7 analytical write-up.

---

## 3. Text-GAN (1D-CNN, word-level) — Generative Inference

### 3a. Decoded generator samples (Part 4 evaluation)

| # | Decoded output |
|---|---|
| 1 | "masterpiece masterpiece masterpiece host host nettelbeck host masterpiece host host nettelbeck nettelbeck roll roll roll roll roll roll roll..." |
| 2 | "masterpiece host host nettelbeck nettelbeck initial host host nettelbeck nettelbeck host host host initial host host host host host host..." |
| 3 | "masterpiece host host host host nettelbeck nettelbeck host host host host initial initial nettelbeck nettelbeck nettelbeck masterpiece..." |
| 4 | "masterpiece host host host host host host initial nettelbeck nettelbeck nettelbeck nettelbeck roll roll roll masterpiece masterpiece host..." |
| 5 | "masterpiece host host host nettelbeck nettelbeck nettelbeck nettelbeck initial host nettelbeck nettelbeck nettelbeck roll masterpiece..." |

### 3b. Inference cell samples, shown directly alongside GPT for contrast (Part 5)

| # | Decoded output |
|---|---|
| 1 | "masterpiece masterpiece masterpiece host host nettelbeck host masterpiece host host nettelbeck nettelbeck roll roll roll roll roll roll roll..." |
| 2 | "masterpiece host host nettelbeck nettelbeck initial host host nettelbeck nettelbeck host host host initial host host host host host host..." |
| 3 | "masterpiece host host host host nettelbeck nettelbeck host host host host initial initial nettelbeck nettelbeck nettelbeck masterpiece..." |

**Aggregate (30 generated samples, best-match BLEU/ROUGE vs. a 200-sentence real-corpus reference pool):** Avg BLEU = 0.0012, Avg ROUGE-1 = 0.0051, Avg ROUGE-L = 0.0051.

**Discriminator classification (256 real vs. 256 generated, held-out):** Accuracy = 0.9961, Precision = 1.0000, Recall = 0.9922, F1 = 0.9961.

*Why the output looks like this:* this run's training history (see `TRAINING_HISTORY.md`) shows the discriminator pulling ahead of the generator for most of the 15 epochs, with only a brief partial recovery by the generator around epoch 11. By the final epoch, the discriminator was again dominant — not fully saturated at 1.0000, but close enough (0.9961 accuracy) that `D(G(z))` is heavily compressed toward 0 for generated samples, sharply weakening the gradient signal reaching the generator. The repeated cycling through a small set of specific words ("masterpiece," "host," "nettelbeck") rather than diverse sentences is the visible symptom of that imbalance. Full diagnosis in the notebook's Part 7.

---

## Side-by-side: GPT vs. Text-GAN on the same task

Both models are asked to produce free-form text without a paired ground-truth target (Part 5's open inference). The contrast is stark:

- **GPT-2:** fluent, grammatical, topically coherent multi-clause sentences.
- **Text-GAN:** long runs of the same handful of nouns/adjectives repeated in different orders, with no real grammatical structure.

This is the empirical core of the notebook's Part 7 discussion: autoregressive transformer-based generation (GPT) versus adversarially-trained discrete-token generation (Text-GAN) under a comparable (lightweight, fine-tuning-scale) compute budget, and what a real (oscillating, discriminator-favored) GAN training curve looks like in practice.