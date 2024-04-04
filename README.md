This is a code repository

# results

|Model              |Learning Rate  |Batch Size |Epochs |TrA    |TrF1   |TeA    |TeF1   |Time       |device   |
|:--:               |:--:           |:--:       |:--:   |:--:   |:--:   |:--:   |:--:   |:--:       |:--:     |
|Bert-base-uncasesd |2e-4           |16         |3      |       |       |0.66   |0.78   |           |M3       |
|Bert-base-uncasesd |2e-4           |16         |4      |       |       |0.6516 |0.7773 |25m54.8s   |RTX4060Ti16G|
|Bert-base-uncasesd |2e-4           |16         |5      |       |       |0.6646 |0.7832 |21m3.1s    |RTX4060Ti16G|
|Bert-base-uncasesd |1e-5           |16         |3      |       |       |0.4160 |0.5503 |17m46.5s   |RTX4060Ti16G|
|Bert-base-uncasesd |5e-5           |16         |3      |       |       |0.6541 |0.7800 |17m56.8s   |RTX4060Ti16G|