import pandas as pd
import numpy as np
import torch
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm import tqdm

# Custom dataset class for handling text data using PyTorch
class MyDataset(Dataset):
    def __init__(self, df):
        super(MyDataset, self).__init__()
        self.data_list = []
        self.end_of_text_token = "  "  # Define a token to signify the end of text
        for _, row in df.iterrows():
            # Create a string for each row and append the end-of-text token
            data_str = f"{row[0]}{self.end_of_text_token}"
            self.data_list.append(data_str)

    def __len__(self):
        # Return the number of items in the dataset
        return len(self.data_list)

    def __getitem__(self, idx):
        # Retrieve an item by index
        return self.data_list[idx]

# Function to train the model
def train(epochs, data_loader, batch_size, tokenizer, model, device):
    model.train()  # Set the model to training mode
    optimizer = AdamW(model.parameters(), lr=1e-3)  # Define the optimizer
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=-1)
    
    for epoch in range(epochs):
        print(f'Running epoch {epoch + 1}')
        for idx, txt in enumerate(data_loader):
            txt_tensor = torch.tensor(tokenizer.encode(txt[0]), device=device).unsqueeze(0)
            outputs = model(txt_tensor, labels=txt_tensor)
            loss = outputs.loss
            loss.backward()

            if (idx + 1) % batch_size == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
    
    return model

# Function to save the trained model to disk
def save_model(model, model_name):
    torch.save(model.state_dict(), f"{model_name}.pt")

# Function to choose the next token from the top probable choices
def choose_from_top_probs(probs, k=50, p=0.8):
    ind = np.argpartition(probs, -k)[-k:]
    top_probs = probs[ind]
    top_probs = {i: top_probs[j] for j, i in enumerate(ind)}
    sorted_probs = sorted(top_probs.items(), key=lambda x: x[1], reverse=True)
    
    cum_prob = np.array([p[1] for p in sorted_probs]).cumsum()
    chosen = cum_prob >= p
    chosen_idx = np.argmax(chosen)
    
    top_indices = [p[0] for p in sorted_probs][:chosen_idx + 1]
    top_probs = [p[1] for p in sorted_probs][:chosen_idx + 1]
    top_probs /= np.sum(top_probs)
    
    return int(np.random.choice(top_indices, p=top_probs))

# Function to generate text using the trained model
def generate_text(tokenizer, model, num_sentences, start_label, device):
    model.eval()  # Set the model to evaluation mode
    results = []
    
    with torch.no_grad():  # Disable gradient calculations
        for _ in tqdm(range(num_sentences)):
            cur_ids = torch.tensor(tokenizer.encode(start_label), device=device).unsqueeze(0)
            for _ in range(100):  # Generate up to 100 tokens
                outputs = model(cur_ids)
                logits = outputs.logits
                softmax_logits = torch.softmax(logits[0, -1], dim=0)
                next_token_id = choose_from_top_probs(softmax_logits.cpu().numpy())
                cur_ids = torch.cat((cur_ids, torch.tensor([[next_token_id]], device=device)), dim=1)
                
                if next_token_id == tokenizer.encode('')[0]:  # Check for end of text token
                    break

            output_seq = list(cur_ids.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_seq).replace(start_label, '')
            results.append(output_text.strip())
    
    return results

# Main execution logic
if __name__ == '__main__':
    dataset_path = '../Data/Train_Dataset.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    df = pd.read_csv(dataset_path)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)
    
    # Prepare datasets and dataloaders
    dataset_sarcastic = MyDataset(df[df['sarcastic'] == 1])
    dataset_non_sarcastic = MyDataset(df[df['sarcastic'] == 0])
    loader_sarcastic = DataLoader(dataset_sarcastic, batch_size=1, shuffle=True)
    loader_non_sarcastic = DataLoader(dataset_non_sarcastic, batch_size=1, shuffle=True)
    
    # Training
    trained_model = train(4, loader_sarcastic, 8, tokenizer, model, device)
    save_model(trained_model, 'sarcastic_model')
    
    # Generation
    sarcastic_texts = generate_text(tokenizer, trained_model, 4000, 'SAR', device)
    with open('SAR_output.txt', 'w') as file:
        for text in sarcastic_texts:
            file.write(text + "\n")
