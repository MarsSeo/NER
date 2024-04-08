This is a code repository

# <center>1. NER Task</center>
## 1.1. BertBase(Classification Fine Tuned) NER results

### Bert-Base Model with Learning Rate 2e-4

| Epochs | Batch Size | Accuracy |   F1   | Recall | Precision |  Time  |   Device    | Framework |
|:------:|:----------:|:--------:|:------:|:------:|:---------:|:------:|:-----------:|:---------:|
|   1    |     16     |   0.56   |  0.70  |  0.56  |   0.95    | 09m23s | M3 Max 30   |  PyTorch  |
|   2    |     16     |   0.69   |  0.80  |  0.69  |   0.95    | 09m12s | M3 Max 30   |  PyTorch  |
|   3    |     16     |   0.56   |  0.71  |  0.56  |   0.96    | 09m15s | M3 Max 30   |  PyTorch  |
|   4    |     16     |   0.64   |  0.77  |  0.64  |   0.96    | 09m23s | M3 Max 30   |  PyTorch  |
|   5    |     16     |   0.65   |  0.78  |  0.65  |   0.95    | 09m39s | M3 Max 30   |  PyTorch  |

### Bert-Base Model with Learning Rate 3e-5

| Epochs | Batch Size | Accuracy |   F1   | Recall | Precision |  Time  |   Device    | Framework |
|:------:|:----------:|:--------:|:------:|:------:|:---------:|:------:|:-----------:|:---------:|
|   1    |     16     |   0.00   |  0.00  |  0.00  |   0.00    | 00 | M3 Max 30   |  PyTorch  |
|   2    |     16     |   0.00   |  0.00  |  0.00  |   0.00    | 00 | M3 Max 30   |  PyTorch  |
|   3    |     16     |   0.00   |  0.00  |  0.00  |   0.00    | 00 | M3 Max 30   |  PyTorch  |
|   4    |     16     |   0.00   |  0.00  |  0.00  |   0.00    | 00 | M3 Max 30   |  PyTorch  |
|   5    |     16     |   0.00   |  0.00  |  0.00  |   0.00    | 00 | M3 Max 30   |  PyTorch  |



## 1.2. Customized BERT-BiLSTM NER results

| Model       | Learning Rate | Batch Size | Epochs | Acc  | F1   | Recall | Precision | Time  | Device      | Framework |
|-------------|---------------|------------|--------|------|------|--------|-----------|-------|-------------|-----------|
| Bert-BiLSTM | 2e-4          | 16         | 1      | 0.58 | 0.71 | 0.58   | 0.94      | 11m5s | M3 Max 30   | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 2      | 0.59 | 0.73 | 0.59   | 0.95      | 11m2s | M3 Max 30   | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 3      | 0.62 | 0.75 | 0.62   | 0.96      | 10m5s | M3 Max 30   | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 4      | 0.69 | 0.80 | 0.69   | 0.95      | 11m5s | M3 Max 30   | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 5      | 0.61 | 0.74 | 0.61   | 0.95      | 11m5s | M3 Max 30   | PyTorch   |


# <center>2. NRE results</center>

| Model         | Learning Rate | Batch Size | Epochs | Acc    | F1(ma) | Recall | Precision |   Time   |     Device      | Framework  |
|:-------------:|:-------------:|:----------:|:------:|:------:|:------:|:------:|:---------:|:--------:|:---------------:|:----------:|
| Bert-Based    |      1e-5     |     16     |   5    | 0.7457 | 0.6718 | 0.6818 |   0.7661  | 10m13.0s | RTX4060Ti16G    | TensorFlow |
| Bert-Base     |      3e-5     |     16     |   3    | 0.7619 | 0.7023 | 0.7113 |   0.7501  |  5m26.5s | RTX4060Ti16G    | TensorFlow |
| Bert-Base     |      3e-5     |     16     |  5 OF  | 0.7637 | 0.7198 | 0.7259 |   0.7693  |  8m59.1s | RTX4060Ti16G    | TensorFlow |
| Bert-Base     |      3e-5     |     16     |  8 OF  | 0.7603 | 0.7077 | 0.7180 |   0.7562  | 15m44.8s | RTX4060Ti16G    | TensorFlow |
| Bert-Large    |      3e-5     |     16     |   3    | 0.7795 | 0.7329 | 0.7453 |   0.7770  | 14m16.8s | RTX4060Ti16G    | TensorFlow |
| Bert-Large*   |      3e-5     |     16     |   5    | 0.7777 | 0.7241 | 0.7382 |   0.7665  | 24m55.3s | RTX4060Ti16G    | TensorFlow |
| RoBERTa-Base  |      3e-5     |     16     |   3    | 0.7575 | 0.7073 | 0.7205 |   0.7502  | 11m57.0s | RTX4060Ti16G    | TensorFlow |
| RoBERTa-Base* |      3e-5     |     16     |   5    | 0.7722 | 0.7246 | 0.7347 |   0.7698  | 21m40.0s | RTX4060Ti16G    | TensorFlow |
| RoBERTA-Large |      3e-5     |     16     |   5    | 0.7835 | 0.7391 | 0.7457 |   0.7879  | 35m29.3s | RTX4060Ti16G    | TensorFlow |


Some small not-syntax bugs exist in NRE problem. Using categorical accuracy.
*Model and its result on testset are saved.
