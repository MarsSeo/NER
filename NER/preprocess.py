import json
import pandas as pd
import torch
from transformers import BertTokenizer

def load_data_to_dataframe(file_path):  
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)

def tokenize_and_align_labels(df, tokenizer, label_map, max_length):
    input_ids = []
    attention_masks = []
    label_ids = []

    for _, row in df.iterrows():
        text = row['token']
        labels = ['O'] * len(text)  

        h_start, h_end = row['h']['pos']
        t_start, t_end = row['t']['pos']
        labels[h_start:h_end] = ['H'] * (h_end - h_start)
        labels[t_start:t_end] = ['T'] * (t_end - t_start)

        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            is_split_into_words=True,
            return_tensors='pt'
        )

        numeric_labels = [label_map[label] for label in labels]
        numeric_labels = [label_map['O']] + numeric_labels[:max_length-2] + [label_map['O']] * (max_length - len(numeric_labels) - 1)

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        label_ids.append(torch.tensor(numeric_labels))

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    label_ids = torch.stack(label_ids, dim=0)
    return input_ids, attention_masks, label_ids
