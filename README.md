# GoEmotions: Fine-Grained Emotion Classification

28-class multi-label emotion classification on Reddit comments using the [GoEmotions dataset](https://huggingface.co/datasets/google-research-datasets/go_emotions) by Google Research.

We progressively build from a bag-of-words baseline to a fine-tuned RoBERTa transformer, comparing performance across four models.

---

## 📊 Results Summary

| Model | Macro F1 |
|---|---|
| Majority-Class Baseline | 0.02 |
| TF-IDF + Logistic Regression | 0.38 |
| BiLSTM + GloVe 300d | 0.37 |
| **RoBERTa Fine-Tuned** | **0.53** |

---

## 🧠 Models

### 1. Majority-Class Baseline
Predicts "neutral" for every example. Scores near zero on 27 out of 28 emotion classes — this is the floor we work from.

### 2. TF-IDF + Logistic Regression
Bag-of-words representation with balanced class weights to handle the heavy class imbalance (neutral dominates with ~18k samples; grief has only ~100). Per-class threshold tuning on the validation set.

### 3. BiLSTM + GloVe 300d
Pre-trained GloVe embeddings (300-dimensional, trained on 6B tokens) fed into a Bidirectional LSTM. Only ~59% GloVe coverage on this Reddit corpus, limiting gains. Threshold tuning applied per class.

### 4. RoBERTa Fine-Tuning
Fine-tuned `roberta-base` via HuggingFace `Trainer`. Uses sqrt-dampened positive class weights in the BCE loss to handle imbalance without exploding gradients. Per-class threshold optimization on validation outputs gives the final classifier.

---

## 📁 Project Structure

```
.
├── NLP_goemotion.ipynb      # Main notebook: all models, training, evaluation & analysis
├── goemotions_1.csv         # GoEmotions dataset (part 1)
├── goemotions_2.csv         # GoEmotions dataset (part 2)
├── goemotions_3.csv         # GoEmotions dataset (part 3)
└── Paper.pdf                # Reference paper
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install datasets transformers torch scikit-learn pandas numpy matplotlib
```

For the BiLSTM model, download [GloVe 300d embeddings](https://nlp.stanford.edu/projects/glove/) and place `glove.6B.300d.txt` in the project directory.

### Running

Open `NLP_goemotion.ipynb` in **Jupyter Notebook** or **Google Colab** and run cells sequentially.

> **Note:** The notebook was developed and tested on Google Colab. For the RoBERTa fine-tuning step, a GPU is strongly recommended.

---

## 📈 Error Analysis Highlights

- **Easiest emotions:** gratitude, amusement (strong keyword signals)
- **Hardest emotions:** realization, disappointment, nervousness (subtle even with full context)
- **Top confusion pairs:** anger↔annoyance, approval↔admiration, confusion↔curiosity
- **Key pattern:** "neutral" acts as catch-all when model confidence is low, absorbing many subtle emotions

---

## 📦 Dataset

The [GoEmotions dataset](https://arxiv.org/abs/2005.00547) consists of 58k Reddit comments annotated with 27 emotion categories + neutral. It is loaded directly via the HuggingFace `datasets` library.

- Train: 43,410 examples
- Val: 5,426 examples
- Test: 5,427 examples

---

## 📄 License

This project is for academic/educational purposes. The GoEmotions dataset is released by Google Research under the [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).
