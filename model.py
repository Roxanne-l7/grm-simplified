import torch
import torch.nn as nn

class GRMModel(nn.Module):
    def __init__(self, num_items, emb_dim, n_heads, n_layers):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out = nn.Linear(emb_dim, num_items)

    def forward(self, x):
        mask = (x == 0)
        x = self.item_emb(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x[:, -1, :]
        return self.out(x)
