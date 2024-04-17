import pandas as pd
import numpy as np
import torch
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm import tqdm
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import random
huggingface_hub.login("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")
# THUDM/chatglm3-6b

# Function to generate text using the trained model
def generate_text(tokenizer, model, num_sentences, start_label, device):
    model.eval()  # Set the model to evaluation mode
    results = []
    def prompt_formatter(examples: List[str], tokenizer) -> str:
        """
        Augments query with text-based context from context_items.
        """
    
        # Create a base prompt with examples to help the model
        # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
        # We could also write this in a txt file and import it in if we wanted.
        base_prompt = """Assuming you are a master of English literature and proficient in writing satirical sentences, 
        here are some example3 for you. Please imitate these sentences and write a new satirical sentence.
        Example1: {:s}
        Example2: {:s}
        Example3: {:s}
        Your Sentence: """
        # Update base prompt with context items and query
        base_prompt = base_prompt.format(examples[0], examples[1], examples[2])
        len_base = len(base_prompt)
        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
            "content": base_prompt}
        ]
    
        # Apply the chat template
        prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                              tokenize=False,
                                              add_generation_prompt=True)
        return prompt, len_base
    
    with torch.no_grad():  # Disable gradient calculations
        for _ in tqdm(range(num_sentences)):
            examples = random.choices(start_label, k=3)
            prompt, len_base = prompt_formatter(examples, tokenizer)
            len_prompt = len(prompt)
            input_ids = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**input_ids,
                                 temperature=0.7,
                                 # lower temperature = more deterministic outputs, higher temperature = more creative outputs
                                 do_sample=True,
                                 # whether or not to use sampling, see https://huyenchip.com/2024/01/16/sampling.html for more
                                 max_new_tokens=100)
            # print("aaa")
            
            output_text = tokenizer.decode(outputs[0])
            # print(output_text)
            output_text = output_text.replace(prompt, '')
            output_text = output_text[len_prompt:]
            # print("BBB")
            # print(prompt)
            # print("CCC")
            # print(output_text)
            tmp = output_text.split("\n")
            tmp = tmp[1:]
            output_text = "\n".join(tmp)
            # print(len_prompt)
            # print(len_base)
            # print(output_text)
            # print(0/0)
            results.append(output_text.strip())
    
    return results

# Main execution logic
if __name__ == '__main__':
    dataset_path = '../Dataset/Train_Dataset.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    df = pd.read_csv(dataset_path)
    
    
    # Prepare datasets and dataloaders
    dataset_sarcastic = df[df['sarcastic'] == 1]
    dataset_non_sarcastic = df[df['sarcastic'] == 0]
    # loader_sarcastic = DataLoader(dataset_sarcastic, batch_size=1, shuffle=True)
    # loader_non_sarcastic = DataLoader(dataset_non_sarcastic, batch_size=1, shuffle=True)
    # print(dataset_sarcastic['tweet'].to_list())
    # print(0/0)
    # Training
    tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('THUDM/chatglm3-6b', trust_remote_code=True).to(device)
    # ChatGLM 我们将直接使用本地模型配合prompt，不再进行微调（微调的显存占用实在过大了）
    
    
    # Generation
    sarcastic_texts = generate_text(tokenizer, model, 4000, dataset_sarcastic['tweet'].to_list(), device)
    with open('output_chatGLM.txt', 'w', encoding='UTF-8') as file:
        for text in sarcastic_texts:
            file.write(text + "\n")
