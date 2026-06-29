# Test Inference Log

Qualitative inference outputs from all three model variants, captured from the run documented in `ACT_4_Lumabi_Manuel_Santillan_ipynb_-_Colab.pdf`. This is a standalone extract — the full code that produced these outputs lives in the notebook's Part 3 (GPT BLEU/ROUGE) and Part 5 (Inference Verification) cells.

---

## 1. BERT — Classification (quantitative, not generative)

BERT has no generation capability — its "inference" is a classification decision (positive/negative sentiment), not free text. Its qualitative output is therefore reported via metrics rather than sample text:

| Metric | Value |
|---|---|
| Precision | 0.8659 |
| Recall | 0.8724 |
| F1 | 0.8692 |
| Validation loss | 0.3183 |

---

## 2. GPT-2 (`distilgpt2`) — Generative Inference

### 2a. Domain-specific prompt completion (Part 5)

**Prompt:** `"This movie was"`

| # | Generated continuation |
|---|---|
| 1 | "This movie was made by a cast that can't seem to have been brought up in a lot of times in the last few years..." |
| 2 | "This movie was so bad it wasn't as bad as it should be, and the actors, along with the acting, acted the way, and..." |
| 3 | "This movie was made by a group of people who have never met a movie before... and who will probably be watching it..." |

Decoding strategy: `do_sample=True, top_k=50, top_p=0.95, temperature=0.7` (Fan et al. 2018 top-k + Holtzman et al. 2019 nucleus sampling).

### 2b. Prompt → ground-truth-continuation comparisons, with BLEU (Part 3)

Each row: a real test-set sentence split into a 4-token prompt + the actual continuation, compared against GPT's own continuation from that same prompt.

| Prompt | Reference continuation | Generated continuation | BLEU |
|---|---|---|---|
| "noyce creates a film" | "of near-hypnotic physical beauty even as he tells a story as horrifying as any in the heart-breakingly extensive ann..." | "that is both intriguing and entertaining, but ultimately lacks the imagination to make it worthwhile..." | 0.0128 |
| "[scherfig] has made a" | "movie that will leave you wondering about the characters' lives after the clever credits roll." | "superb film about a woman's career in life, and it's also an excellent example of how women make their" | 0.0130 |
| "the dramatic scenes are" | "frequently unintentionally funny, and the action sequences -- clearly the main event -- are surprisingly uninvolvin..." | "very much like the last few scenes of the first half of the movie. the performances are much more like the last few" | 0.0139 |

**Aggregate (29 scored test samples):** Avg BLEU = 0.0174, Avg ROUGE-1 = 0.1043, Avg ROUGE-L = 0.0875.

*Note on low BLEU despite fluent output:* BLEU/ROUGE penalize any lexical divergence from one specific reference continuation. The generated text above is grammatical and topically on-point, but uses different words/phrasing than the one ground-truth continuation it's scored against — this is expected and is discussed further in the notebook's Part 7 analytical write-up.

---

## 3. Text-GAN (1D-CNN, word-level) — Generative Inference

### 3a. Decoded generator samples (Part 4 evaluation)

| # | Decoded output |
|---|---|
| 1 | "infuriating pacing" |
| 2 | "infuriating infuriating" |
| 3 | "infuriating infuriating" |
| 4 | "tedious filme filme filme filme filme filme filme difference infuriating" |
| 5 | "filme heady infuriating infuriating" |

### 3b. Inference cell samples, shown directly alongside GPT for contrast (Part 5)

| # | Decoded output |
|---|---|
| 1 | "infuriating pacing" |
| 2 | "infuriating infuriating" |
| 3 | "infuriating infuriating" |

**Aggregate (30 generated samples, best-match BLEU/ROUGE vs. a 200-sentence real-corpus reference pool):** Avg BLEU = 0.0002, Avg ROUGE-1 = 0.0034, Avg ROUGE-L = 0.0034.

**Discriminator classification (256 real vs. 256 generated, held-out):** Accuracy = Precision = Recall = F1 = 1.0000.

*Why the output looks like this:* the discriminator's perfect score means it has completely separated real from fake — this saturates `D(G(z))` near 0 for every generated sample, which starves the generator of a useful training gradient (a documented GAN pathology; see Goodfellow et al. 2014). The repetition of a narrow set of specific words ("infuriating", "filme") rather than diverse sentences is the visible symptom of that stalled training. Full diagnosis in the notebook's Part 7.

---

## Side-by-side: GPT vs. Text-GAN on the same task

Both models are asked to produce free-form text without a paired ground-truth target (Part 5's open inference). The contrast is stark:

- **GPT-2:** fluent, grammatical, topically coherent multi-clause sentences.
- **Text-GAN:** short, repetitive, often ungrammatical fragments dominated by a handful of tokens.

This is the empirical core of the notebook's Part 7 discussion: autoregressive transformer-based generation (GPT) versus adversarially-trained discrete-token generation (Text-GAN) under a comparable (lightweight, fine-tuning-scale) compute budget.
