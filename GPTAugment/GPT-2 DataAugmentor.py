import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
import gc
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, dataframe):
        super(TextDataset, self).__init__()
        self.data = [f"{row[0]}  " for _, row in dataframe.iterrows()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_model(num_epochs, loader, batch_sz, tokenizer, model, device):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        total_loss = 0
        model.train()
        
        for idx, text in enumerate(loader):
            encoded_input = torch.tensor(tokenizer.encode(text[0])).unsqueeze(0).to(device)
            outputs = model(encoded_input, labels=encoded_input)
            loss = outputs.loss
            loss.backward()
            
            if idx % batch_sz == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
                print(f"Batch {idx // batch_sz + 1}, Loss: {loss.item()}")
                
        print(f"Total loss after epoch {epoch+1}: {total_loss}")

    return model

def save_model(model, filename):
    torch.save(model.state_dict(), f"{filename}.pt")

def generate_text(tokenizer, model, num_sentences, start_text):
    model.eval()
    sentences = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_sentences)):
            generated = torch.tensor(tokenizer.encode(start_text)).unsqueeze(0).to(device)
            
            for __ in range(100):  # Limiting to 100 tokens
                outputs = model(generated)
                predictions = outputs.logits[:, -1, :]
                predicted_index = torch.argmax(predictions, axis=-1)
                generated = torch.cat((generated, predicted_index.unsqueeze(0)), dim=1)
                
                if predicted_index == tokenizer.encode('')[0]:
                    break

            decoded_sequence = tokenizer.decode(generated.tolist()[0])
            sentences.append(decoded_sequence.replace(start_text, ''))
    
    return sentences

if __name__ == "__main__":
    data_path = 'data/Train_Dataset.csv'
    df = pd.read_csv(data_path)
    sarcastic_df = df[df['sarcastic'] == 1]
    not_sarcastic_df = df[df['sarcastic'] == 0]

    sarcastic_dataset = TextDataset(sarcastic_df)
    not_sarcastic_dataset = TextDataset(not_sarcastic_df)
    
    sarcastic_loader = DataLoader(sarcastic_dataset, batch_size=1, shuffle=True)
    not_sarcastic_loader = DataLoader(not_sarcastic_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)
    
    gc.collect()
    torch.cuda.empty_cache()

    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=-1)

    trained_model = train_model(4, sarcastic_loader, 8, tokenizer, model, device)
    save_model(trained_model, 'trained_sarcastic_model')
    
    generated_sentences = generate_text(tokenizer, model, 100, "Example start text")
    print(generated_sentences)
