import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GRMDataset, MAX_SEQ_LEN
from model import GRMModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, EPOCHS = 128, 64, 2, 2, 3

def load_data(path):
    df = pd.read_csv(path, sep='::', engine='python',
                     names=['user_id', 'item_id', 'rating', 'timestamp'])
    df = df[df['rating'] >= 4.0].sort_values(['user_id', 'timestamp'])
    user2id = {uid: i for i, uid in enumerate(df['user_id'].unique())}
    item2id = {iid: i+1 for i, iid in enumerate(df['item_id'].unique())}
    df['user_id'] = df['user_id'].map(user2id)
    df['item_id'] = df['item_id'].map(item2id)
    num_items = len(item2id) + 1
    user_seq = df.groupby('user_id')['item_id'].apply(list)
    user_seq = user_seq.sample(n=2000, random_state=42)
    return user_seq.tolist(), num_items

def train():
    sequences, num_items = load_data('/kaggle/input/movielens-1m-dataset/ratings.dat')
    dataset = GRMDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = GRMModel(num_items, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in tqdm(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE).squeeze()
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), "grm_model.pth")
    return model, dataloader

if __name__ == "__main__":
    train()
