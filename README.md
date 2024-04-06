This is a code repository

# BERT BASE NER results

|Model              |Learning Rate  |Batch Size |Epochs |TrA    |TrF1   |TeA    |TeF1   |Time       |device   |framework|
|:--:               |:--:           |:--:       |:--:   |:--:   |:--:   |:--:   |:--:   |:--:       |:--:     |:--:     |
|Bert-base-uncasesd |2e-4           |16         |3      |       |       |0.66   |0.78   |           |M3       |pytorch  |
|Bert-base-uncasesd |2e-4           |16         |4      |       |       |0.6516 |0.7773 |25m54.8s   |RTX4060Ti16G|pytorch|
|Bert-base-uncasesd |2e-4           |16         |5      |       |       |0.6646 |0.7832 |21m3.1s    |RTX4060Ti16G|pytorch|
|Bert-base-uncasesd |1e-5           |16         |3      |       |       |0.4160 |0.5503 |17m46.5s   |RTX4060Ti16G|pytorch|
|Bert-base-uncasesd |5e-5           |16         |3      |       |       |0.6541 |0.7800 |17m56.8s   |RTX4060Ti16G|pytorch|

# Customized BERT-BiLSTM NER results
|Model      |Learning Rate  |Batch Size |Epochs |TrA    |TrF1   |TeA    |TeF1   |Time       |device   |framework|
|:--:       |:--:           |:--:       |:--:   |:--:   |:--:   |:--:   |:--:   |:--:       |:--:     |:--:     |
|Bert-BiLSTM|2e-4           |16         |3      |       |       |       |       |           |M3       |pytorch  |
|Bert-BiLSTM|2e-4           |16         |4      |       |       |       |       |           |M3       |pytorch  |
|Bert-BiLSTM|2e-4           |16         |5      |       |       |       |       |           |M3       |pytorch  |
|Bert-BiLSTM|1e-5           |16         |3      |       |       |       |       |           |M3       |pytorch  |
|Bert-BiLSTM|5e-5           |16         |3      |       |       |       |       |           |M3       |pytorch  |

# NRE results

|Model              |Learning Rate  |Batch Size |Epochs |TrA(MAX)|TrF1   |TeA    |TeF1   |Time       |device   |framework|
|:--:               |:--:           |:--:       |:--:   |:--:   |:--:   |:--:   |:--:   |:--:       |:--:     |:--:     |
|Bert-base-uncasesd |2e-5           |16         |3      |0.7221 |       |0.6408 |       |5m26.3s|RTX4060Ti16G|tensorflow|

Some small not-syntax bugs exist in NRE problem. Using categorical accuracy.
