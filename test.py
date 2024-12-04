from typing import List, Dict, Optional
from base import BaseRouter, BaseRetriever, BaseReflection, BaseResponder, Message
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


class LangchainReflection():
    def __init__(self, llm):
        prompt = ChatPromptTemplate.from_messages([
            # ("system", "Bạn là"),
            # MessagesPlaceholder(variable_name="chat_history"),
            ("human", contextualize_q_system_prompt)
        ])
        
        self.chain = prompt | llm | StrOutputParser()

    def expand_query(self, question: str, chat_history, num_context: int = 2) -> str:
        # formatted_history = [
        #     HumanMessage(content=msg.content) if msg.type == 'human'
        #     else AIMessage(content=msg.content)
        #     for msg in chat_history[-num_context*2:]
        # ]
        # formatted_history = []
        # for msg in chat_history[-num_context*2:]:
        #     if msg.type == 'human':
        #         formatted_history.append(msg.content)
           
        # print('\n'.join(formatted_history))
        res = self.chain.invoke({
            "question": question,
            "chat_history": '\n'.join(chat_history)
        })
        return res

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)

contextualize_q_system_prompt = """Cho câu hỏi trước và câu hỏi mới, hãy viết lại câu hỏi để hiểu mà không cần sử dụng câu hỏi trước đó 
Câu hỏi trước: {prev}
Câu hỏi mới: {present}
Lưu ý: Chỉ viết lại nếu THỰC SỰ cần thiết (câu hỏi có liên quan đến ngữ cảnh câu trước đó) , nếu không hãy in lại câu hỏi và không sửa đổi gì thêm
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", contextualize_q_system_prompt),
    ]
)

chain = contextualize_q_prompt | llm | StrOutputParser()

chat_hístory = [
]

print (chain.invoke({
    "prev": "",
    "present": "muốn học kỹ sư thì phải đăng ký như nào"
}))

# a = LangchainReflection(llm)
# print(a.expand_query("", ["human: Muốn học chương trình kỹ sư thì làm như thế nào"], 2))


import torch
from torch import nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

class ContrastiveDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        anchor = self.tokenizer(row['anchor'], 
                              padding='max_length',
                              truncation=True,
                              max_length=self.max_length,
                              return_tensors='pt')
        
        positive = self.tokenizer(row['positive'],
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors='pt')
        
        negative = self.tokenizer(row['negative'],
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors='pt')

        return {
            'anchor_input_ids': anchor['input_ids'].squeeze(),
            'anchor_attention_mask': anchor['attention_mask'].squeeze(),
            'positive_input_ids': positive['input_ids'].squeeze(),
            'positive_attention_mask': positive['attention_mask'].squeeze(),
            'negative_input_ids': negative['input_ids'].squeeze(),
            'negative_attention_mask': negative['attention_mask'].squeeze(),
        }

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, anchor, positive, negative):
        pos_sim = self.cos(anchor, positive)
        neg_sim = self.cos(anchor, negative)
        losses = torch.relu(self.margin - (pos_sim - neg_sim))
        return losses.mean()

def train_model(train_dataloader, model, loss_fn, optimizer, device, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}')
        
        for batch in progress_bar:
            # Move batch to device
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)

            # Get embeddings
            anchor_emb = model(input_ids=anchor_input_ids, 
                             attention_mask=anchor_attention_mask).pooler_output
            positive_emb = model(input_ids=positive_input_ids,
                               attention_mask=positive_attention_mask).pooler_output
            negative_emb = model(input_ids=negative_input_ids,
                               attention_mask=negative_attention_mask).pooler_output

            # Calculate loss
            loss = loss_fn(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
            
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')

def main():
    # Hyperparameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 512
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and tokenizer
    model_name = "BAAI/bge-m3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    
    # Prepare dataset
    dataset = ContrastiveDataset('your_data.csv', tokenizer, MAX_LENGTH)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize loss and optimizer
    loss_fn = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    train_model(train_dataloader, model, loss_fn, optimizer, DEVICE, NUM_EPOCHS)
    
    # Save the fine-tuned model
    model.save_pretrained("bge-m3-finetuned")
    tokenizer.save_pretrained("bge-m3-finetuned")

if __name__ == "__main__":
    main()