This is a code repository

# 1. NER Task
## 1.1. BertBase(Classification Fine Tuned) NER results

| Model | Learning Rate | Batch Size | Epochs | Acc  | F1   | Recall | Precision | Time  | Device     | Framework |
|-------|---------------|------------|--------|------|------|--------|-----------|-------|------------|-----------|
| Bert  | 2e-4          | 16         | 1      | 0.56 | 0.70 | 0.56   | 0.95      | 09m23s | M3 Max 30  | PyTorch   |
| Bert  | 2e-4          | 16         | 2      | 0.69 | 0.80 | 0.69   | 0.95      | 09m12s | M3 Max 30  | PyTorch   |
| Bert  | 2e-4          | 16         | 3      | 0.56 | 0.71 | 0.56   | 0.96      | 09m15s | M3 Max 30  | PyTorch   |
| Bert  | 2e-4          | 16         | 4      | 0.64 | 0.77 | 0.64   | 0.96      | 09m23s | M3 Max 30  | PyTorch   |
| Bert  | 2e-4          | 16         | 5      | 0.65 | 0.78 | 0.65   | 0.95      | 09m39  | M3 Max 30  | PyTorch   |


## 1.2. Customized BERT-BiLSTM NER results

| Model       | Learning Rate | Batch Size | Epochs | Acc  | F1   | Recall | Precision | Time  | Device      | Framework |
|-------------|---------------|------------|--------|------|------|--------|-----------|-------|-------------|-----------|
| Bert-BiLSTM | 2e-4          | 16         | 1      | 0.58 | 0.71 | 0.58   | 0.94      | 11m5s | M3 Max 30   | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 2      | 0.59 | 0.73 | 0.59   | 0.95      | 11m2s | M3 Max 30   | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 3      | 0.62 | 0.75 | 0.62   | 0.96      | 10m5s | M3 Max 30   | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 4      | 0.69 | 0.80 | 0.69   | 0.95      | 11m5s | M3 Max 30   | PyTorch   |
| Bert-BiLSTM | 2e-4          | 16         | 5      | 0.61 | 0.74 | 0.61   | 0.95      | 11m5s | M3 Max 30   | PyTorch   |


# 2. NRE results

|Model              |Learning Rate  |Batch Size |Epochs |TrA(MAX)|TrF1   |TeA    |TeF1   |Time       |device   |framework|
|:--:               |:--:           |:--:       |:--:   |:--:   |:--:   |:--:   |:--:   |:--:       |:--:     |:--:     |
|Bert-base-uncasesd |2e-5           |16         |3      |0.7221 |       |0.6408 |       |5m26.3s    |RTX4060Ti16G|tensorflow|
|Bert-large-uncasesd|2e-5           |16         |3      |0.8142 |       |0.7063 |       |14m1.3s    |RTX4060Ti16G|tensorflow|

Some small not-syntax bugs exist in NRE problem. Using categorical accuracy.
