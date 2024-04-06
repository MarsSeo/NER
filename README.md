This is a code repository

# 1. NER Task
## 1.1. BERT BASE NER results

|Model              |Learning Rate  |Batch Size |Epochs |TrA    |TrF1   |TeA    |TeF1   |Time       |device   |framework|
|:--:               |:--:           |:--:       |:--:   |:--:   |:--:   |:--:   |:--:   |:--:       |:--:     |:--:     |
|Bert-base-uncasesd |2e-4           |16         |3      |       |       |0.66   |0.78   |           |M3 Max 30 Cores|pytorch  |
|Bert-base-uncasesd |2e-4           |16         |4      |       |       |0.6516 |0.7773 |25m54.8s   |RTX4060Ti16G|pytorch|
|Bert-base-uncasesd |2e-4           |16         |5      |       |       |0.6646 |0.7832 |21m3.1s    |RTX4060Ti16G|pytorch|
|Bert-base-uncasesd |1e-5           |16         |3      |       |       |0.4160 |0.5503 |17m46.5s   |RTX4060Ti16G|pytorch|
|Bert-base-uncasesd |5e-5           |16         |3      |       |       |0.6541 |0.7800 |17m56.8s   |RTX4060Ti16G|pytorch|

## 1.2. Customized BERT-BiLSTM NER results

| Model       | Learning Rate | Batch Size | Epochs | Acc  | F1   | Recall | Precision | Time | device      |
|-------------|---------------|------------|--------|------|------|--------|-----------|------|-------------|
| B-BiLSTM | 2e-4          | 16         | 1      | 0.58 | 0.71 | 0.58   | 0.94      | 11m5s| M3 Max 30     |
| B-BiLSTM | 2e-4          | 16         | 2      | 0.59 | 0.73 | 0.59   | 0.95      | 11m2s| M3 Max 30   |
| B-BiLSTM | 2e-4          | 16         | 3      |      |      |        |           |      | M3 Max 30   |
| B-BiLSTM | 1e-5          | 16         | 4      |      |      |        |           |      | M3 Max 30   |
| B-BiLSTM | 5e-5          | 16         | 5      |      |      |        |           |      | M3 Max 30   | 





# 2. NRE results

|Model              |Learning Rate  |Batch Size |Epochs |TrA(MAX)|TrF1   |TeA    |TeF1   |Time       |device   |framework|
|:--:               |:--:           |:--:       |:--:   |:--:   |:--:   |:--:   |:--:   |:--:       |:--:     |:--:     |
|Bert-base-uncasesd |2e-5           |16         |3      |0.7221 |       |0.6408 |       |5m26.3s    |RTX4060Ti16G|tensorflow|
|Bert-large-uncasesd|2e-5           |16         |3      |0.8142 |       |0.7063 |       |14m1.3s    |RTX4060Ti16G|tensorflow|

Some small not-syntax bugs exist in NRE problem. Using categorical accuracy.
