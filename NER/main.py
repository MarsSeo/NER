import argparse
from models import BertEntityRecognizer
from preprocess import load_data_to_dataframe, tokenize_and_align_labels
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AdamW
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def train(model, dataloader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_labels = batch

            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")

def test(model, dataloader, device):
    model.eval()
    all_predictions, all_true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_labels = batch
            
            outputs = model(b_input_ids, attention_mask=b_attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.flatten().tolist())
            all_true_labels.extend(b_labels.flatten().tolist())

    # Calculate metrics
    precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description='BERT Entity Recognition Trainer and Tester')
    parser.add_argument('mode', type=str, choices=['train', 'test'], help='Mode: train or test')
    args = parser.parse_args()

    config = {
        'MODEL_NAME': 'bert-base-uncased',
        # 'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
        'DEVICE': 'mps',
        'MAX_LENGTH': 512,
        'BATCH_SIZE': 16,
        'EPOCHS': 3,
        'LEARNING_RATE': 2e-5,
        'LABEL_MAP': {'O': 0, 'H': 1, 'T': 2},
        'NUM_LABELS': 3
    }

    tokenizer = BertTokenizer.from_pretrained(config['MODEL_NAME'])
    print('--Load Model Started --')
    model = BertEntityRecognizer(config['MODEL_NAME'], config['NUM_LABELS']).to(config['DEVICE'])
    print(f'--Load Model Finished, Model Name: {config["MODEL_NAME"]} --')
    optimizer = AdamW(model.parameters(), lr=config['LEARNING_RATE'])


    train_df = load_data_to_dataframe('semeval_train.txt')
    input_ids, attention_masks, label_ids = tokenize_and_align_labels(train_df, tokenizer, config['LABEL_MAP'], config['MAX_LENGTH'])
    dataset = TensorDataset(input_ids, attention_masks, label_ids)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=config['BATCH_SIZE'])

    if args.mode == 'train':
        train(model, dataloader, optimizer, config['DEVICE'], config['EPOCHS'])
    elif args.mode == 'test':
        test_dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=config['BATCH_SIZE'])
        test(model, test_dataloader, config['DEVICE'])

if __name__ == '__main__':
    main()
