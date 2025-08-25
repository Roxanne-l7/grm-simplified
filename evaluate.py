import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader, topk=10):
    model.eval()
    hits, ndcgs = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE).squeeze()
            logits = model(x)
            topk_items = torch.topk(logits, topk)[1]
            for pred, true in zip(topk_items, y):
                hits.append(int(true in pred))
                if true in pred:
                    idx = (pred == true).nonzero(as_tuple=True)[0].item()
                    ndcgs.append(1 / np.log2(idx + 2))
                else:
                    ndcgs.append(0)
    print(f"Hit@{topk}: {np.mean(hits):.4f}, NDCG@{topk}: {np.mean(ndcgs):.4f}")
