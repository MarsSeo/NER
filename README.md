# 1. Introduction

## 1.1. Objective
This project aims to explore the capabilities of advanced NLP models in relation extraction tasks, utilizing the SemEval dataset. We focus on training and comparing the performance of three models: BERT, BERT-BiLSTM, and RoBERTa, to understand their effectiveness in understanding and extracting complex relationships within text.

## 1.2. Importance
Relation extraction is a crucial component in the realm of Natural Language Processing. It enables the understanding and mapping of semantic relationships between entities in a text, which is fundamental for various applications such as information retrieval, knowledge graph construction, and data mining.

## 1.3. Overview of Models
BERT (Bidirectional Encoder Representations from Transformers): Known for its deep bidirectional nature, BERT sets a new standard in NLP for understanding context and semantics.
BERT-BiLSTM: Our customized model enhances BERT with a Bi-directional Long Short-Term Memory (BiLSTM) layer, aiming to capture sequential data more effectively, especially in long sentences.
RoBERTa (Robustly Optimized BERT Pretraining Approach): An optimized version of BERT, RoBERTa modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates.

# 2. Background
## 2.1. SemEval Dataset Overview
<img width="836" alt="image" src="https://github.com/MarsSeo/NER/assets/103374757/0355e98d-61ce-4022-8e30-26c50cab19a7">

The SemEval dataset is a rich and structured collection designed for training and evaluating relation extraction models. It includes:

semeval_rel2id.json: This file maps various relationships to their indices. Each relationship is listed twice due to the varying positions of two entities, 
leading to a total of 19 unique relationships including the 'Other' category. semeval_train.txt & semeval_val.txt:
These files split the original training set into training (6507 samples) and validation (1493 samples) subsets, formatted in JSON, and group samples by their relationships.
semeval_test.txt: Contains 2717 samples in the same format as the training and validation sets.<br>

## 2.2. Hardware and Software Specifications
Hardware Configurations:<br>
CPUs: M3 MAX 16 Core, Intel Core i5-12400F, AMD Ryzen 7 6800H<br>
GPUs: M3 MAX 30, RTX 4060 Ti AD 16G, T4 GPU (Google Colab)<br>
Software Configurations:<br>
Operating Systems: Windows, Mac; ;Programming Language: Python<br>
Frameworks: PyTorch, TensorFlow, Transformers<br>


## 3. Novel Approaches
## 3.1. Customized Model - BERT-BiLSTM
The BERT-BiLSTM model is a novel approach in our study. We append a bidirectional LSTM layer to the last layer of the BERT model. This integration aims to leverage both the deep contextual understanding of BERT and the sequential data processing capability of BiLSTM. It is particularly designed to improve relation extraction from longer sentences where context dispersion is common.

## 3.2. Evaluation Metrics
We employ precision, recall, and F1-score to evaluate the model's performance. These metrics are crucial for understanding the balance between accurately identifying relationships and the model's ability to detect as many relevant relations as possible.

# 4. NER Results

![image](https://github.com/MarsSeo/NER/assets/103374757/64e875da-9b1b-412a-9e35-5557af9c57ca)

## 4.1. BertBase(Classification Fine Tuned) NER results

### Bert-Base Model with Learning Rate 2e-4

#### (1) Batch Size: 16

| Epochs | F1   | Recall | Precision |  Time  |   Device    | Framework |
|:------:|:----:|:------:|:---------:|:------:|:-----------:|:---------:|
|   1    | 0.70 |  0.56  |   0.95    | 09m23s | M3 Max 30   |  PyTorch  |
|   2    | 0.80 |  0.69  |   0.95    | 09m12s | M3 Max 30   |  PyTorch  |
|   3    | 0.71 |  0.56  |   0.96    | 09m15s | M3 Max 30   |  PyTorch  |
|   4    | 0.77 |  0.64  |   0.96    | 09m23s | M3 Max 30   |  PyTorch  |
|   5    | 0.78 |  0.65  |   0.95    | 09m39s | M3 Max 30   |  PyTorch  |

#### (2) Batch Size: 8

| Epochs | F1   | Recall | Precision |  Time  | Device  | Framework |
|:------:|:----:|:------:|:---------:|:------:|:-------:|:---------:|
|   1    | 0.69 |  0.56  |   0.95    | 10m17s | T4 GPU  |  PyTorch  |
|   2    | 0.74 |  0.61  |   0.95    | 10m30s | T4 GPU  |  PyTorch  |
|   3    | 0.73 |  0.61  |   0.95    | 10m28s | T4 GPU  |  PyTorch  |

### Bert-Base Model with Learning Rate 3e-5

| Epochs | Batch Size | F1   | Recall | Precision |  Time  |   Device    | Framework |
|:------:|:----------:|:----:|:------:|:---------:|:------:|:-----------:|:---------:|
|   1    |     16     | 0.53 |  0.39  |   0.92    | 10m34s | M3 Max 30   |  PyTorch  |
|   2    |     16     | 0.62 |  0.47  |   0.95    | 10m23s | M3 Max 30   |  PyTorch  |
|   3    |     16     | 0.74 |  0.61  |   0.96    | 10m17s | M3 Max 30   |  PyTorch  |
|   4    |     16     | 0.80 |  0.69  |   0.96    | 19m57s | M3 Max 30   |  PyTorch  |
|   5    |     16     | 0.82 |  0.72  |   0.71    | 11m25s | M3 Max 30   |  PyTorch  |



## 4.2. Customized BERT-BiLSTM NER results

| Model       | Learning Rate | Batch Size | Epochs | F1   | Recall | Precision | Time  | Device    | Framework |
|-------------|---------------|------------|--------|------|--------|-----------|-------|-----------|-----------|
| Bert-BiLSTM | 2e-4          | 16         | 1      | 0.71 | 0.58   | 0.94      | 11m5s | M3 Max 30 | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 2      | 0.73 | 0.59   | 0.95      | 11m2s | M3 Max 30 | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 3      | 0.75 | 0.62   | 0.96      | 10m5s | M3 Max 30 | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 4      | 0.80 | 0.69   | 0.95      | 11m5s | M3 Max 30 | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 5      | 0.74 | 0.61   | 0.95      | 11m5s | M3 Max 30 | PyTorch   |

## 4.3. RoBERTa-base NER results with MAXLENGTH 128

| Model        | Learning Rate | Batch Size | Epochs | F1     | Recall | Precision | Time  | Device       | Framework |
|--------------|---------------|------------|--------|--------|--------|-----------|-------|--------------|-----------|
| RoBERTa-base | 2e-4          | 16         | 1      | 0.6748 | 0.5223 | 0.9586    | 1m8s  | RTX4060Ti16G | PyTorch   |
| RoBERTa-base | 2e-4          | 16         | 2      | 0.7643 | 0.6368 | 0.9598    | 1m5s  | RTX4060Ti16G | PyTorch   |
| RoBERTa-base | 2e-4          | 16         | 3      | 0.7310 | 0.5882 | 0.9671    | 1m5s  | RTX4060Ti16G | PyTorch   |
| RoBERTa-base | 2e-4          | 16         | 4      | 0.7702 | 0.6435 | 0.9631    | 1m5s  | RTX4060Ti16G | PyTorch   |
| RoBERTa-base | 2e-4          | 16         | 5      | 0.7546 | 0.6230 | 0.9579    | 1m5s  | RTX4060Ti16G | PyTorch   |



# 5. NRE (Relation Extraction) Results

![image](https://github.com/MarsSeo/NER/assets/103374757/0ce2cbec-004a-4a94-b689-cf0661abc42c)


| Model         | Learning Rate | Batch Size | Epochs | F1(ma) | Recall | Precision |   Time   |     Device      | Framework  |
|:-------------:|:-------------:|:----------:|:------:|:------:|:------:|:---------:|:--------:|:---------------:|:----------:|
| Bert-Based    |      1e-5     |     16     |   5    | 0.6718 | 0.6818 |   0.7661  | 10m13.0s | RTX4060Ti16G    | TensorFlow |
| Bert-Base     |      3e-5     |     16     |   3    | 0.7023 | 0.7113 |   0.7501  |  5m26.5s | RTX4060Ti16G    | TensorFlow |
| Bert-Base     |      3e-5     |     16     |  5 OF  | 0.7198 | 0.7259 |   0.7693  |  8m59.1s | RTX4060Ti16G    | TensorFlow |
| Bert-Base     |      3e-5     |     16     |  8 OF  | 0.7077 | 0.7180 |   0.7562  | 15m44.8s | RTX4060Ti16G    | TensorFlow |
| Bert-Large    |      3e-5     |     16     |   3    | 0.7329 | 0.7453 |   0.7770  | 14m16.8s | RTX4060Ti16G    | TensorFlow |
| Bert-Large*   |      3e-5     |     16     |   5    | 0.7241 | 0.7382 |   0.7665  | 24m55.3s | RTX4060Ti16G    | TensorFlow |
| RoBERTa-Base  |      3e-5     |     16     |   3    | 0.7073 | 0.7205 |   0.7502  | 11m57.0s | RTX4060Ti16G    | TensorFlow |
| RoBERTa-Base* |      3e-5     |     16     |   5    | 0.7246 | 0.7347 |   0.7698  | 21m40.0s | RTX4060Ti16G    | TensorFlow |
| RoBERTA-Large |      3e-5     |     16     |   5    | 0.7293 | 0.7463 |   0.7703  | 19m47.1s | RTX4060Ti16G    | TensorFlow |
| RoBERTA-Large |      3e-5     |     16     |   5    | 0.7391 | 0.7457 |   0.7879  | 35m29.3s | RTX4060Ti16G    | TensorFlow |
*Model and its result on testset are saved.

<img width="409" alt="image" src="https://github.com/MarsSeo/NER/assets/103374757/2b69c567-84eb-4bef-a280-7613980230a7">
Confusion Matrix: RoBERTa-large epoch=3, lr=3e-5

| Model         |PreMethod| Learning Rate | Batch Size | Epochs | Acc    | F1(ma) | Recall | Precision |   Time   |     Device      | Framework  |
|:-------------:|:---:|:-------------:|:----------:|:------:|:------:|:------:|:------:|:---------:|:--------:|:---------------:|:----------:|
| Bert-Base    |None    |      3e-5     |     16     |   8    | 0.7744 | 0.7301 | 0.7330 |   0.7855  | 14m16.1s | RTX4060Ti16G    | TensorFlow |
| Bert-Base    |Replace |      3e-5     |     16     |   8    | 0.7656 | 0.7116 | 0.7154 |   0.7639  | 14m17.9s | RTX4060Ti16G    | TensorFlow |
| Bert-Base    |Marking |      3e-5     |     16     |   8    | 0.8436 | 0.8044 | 0.8237 |   0.8422  | 14m9.7s | RTX4060Ti16G    | TensorFlow |
| Bert-Base    |MarkingDiff |      3e-5     |     16     |   8    | 0.8432 | 0.8023 | 0.8159 |   0.8441  | 14m12.4s | RTX4060Ti16G    | TensorFlow |
| Bert-Base     |MarkingEvery |      3e-5     |     16     |   8    | 0.8435 | 0.7995 | 0.8083 |   0.8454  | 14m25.2s | RTX4060Ti16G    | TensorFlow |

After optimized

| Model         |PreMethod| Learning Rate | Batch Size | Epochs | Acc    | F1(ma) | Recall | Precision |   Time   |     Device      | Framework  |
|:-------------:|:---:|:-------------:|:----------:|:------:|:------:|:------:|:------:|:---------:|:--------:|:---------------:|:----------:|
| Bert-Base     |Marking |      3e-5     |     16     |   8    | 0.8354 | 0.7902 | 0.7966 |   0.8406  | 14m9.7s | RTX4060Ti16G    | TensorFlow |
| Bert-Large    |Marking |      3e-5     |     16     |   8    | 0.8561 | 0.8172 | 0.8319 |   0.8572  | 38m38.8s | RTX4060Ti16G    | TensorFlow |
| Bert-Base-BiLSTM|Marking |      3e-5     |     16     |   8    | 0.8425 | 0.7993 | 0.8079 |   0.8461  | 15m2.3s | RTX4060Ti16G    | TensorFlow |
| Bert-Large-BiLSTM|Marking |      3e-5     |     16     |   8    | 0.8454 | 0.8070 | 0.8191 |   0.8500  | 41m18.3s | RTX4060Ti16G    | TensorFlow |
| RoBERTa-Base    |Marking |      3e-5     |     16     |   8    | 0.8447 | 0.8077 | 0.8175 |   0.8522  | 14m25.2s | RTX4060Ti16G    | TensorFlow |
| RoBERTa-Large   |Marking |      3e-5     |     16     |   8    | 0.8535 | 0.8130 | 0.8210 |   0.8596  | 58m15.5s | RTX4060Ti16G    | TensorFlow |