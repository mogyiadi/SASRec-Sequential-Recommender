import torch
import numpy as np
import pandas as pd
from SASRec import SASRec, SASRecConfig
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class SasEvalDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        seq = sample["input"]
        target = sample["target"]

        seq = seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(seq)
        seq = [0] * pad_len + seq

        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def evaluate(model, dataLoader, num_items, device, k_list=[10, 20]):
    model.eval()

    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    with torch.no_grad():
        for seq, target in tqdm(dataLoader, desc='Evaluating'):
            seq = seq.to(device)
            target_item = target.item()

            negatives = []
            while len(negatives) < 99:
                n = np.random.randint(1, num_items)
                if n != target_item and n not in negatives:
                    negatives.append(n)

            candidates = [target_item] + negatives
            candidates = torch.tensor(candidates, dtype=torch.long).to(device)

            scores = model.predict(seq, candidates.unsqueeze(0))

            _, top_idxs = torch.sort(scores.squeeze(0), descending=True)

            rank = top_idxs.tolist().index(0) + 1

            for k in k_list:
                if rank <= k:
                    recalls[k].append(1)
                    ndcgs[k].append(1 / np.log2(rank + 1))
                else:
                    recalls[k].append(0)
                    ndcgs[k].append(0)

    final_recall = {k: np.mean(v) for k, v in recalls.items()}
    final_ndcg = {k: np.mean(v) for k, v in ndcgs.items()}

    return final_recall, final_ndcg


if __name__ == "__main__":
    # Load test data
    with open('test.json', 'r') as f:
        test_data = json.load(f)

    with open('train.json', 'r') as f:
        train_data = json.load(f)

    with open('val.json', 'r') as f:
        val_data = json.load(f)

    all_data = train_data + val_data + test_data
    num_items = max(item for d in all_data for item in d['input'] + [d['target']]) + 1

    print(f'Number of items: {num_items}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in ["A", "B", "C"]:
        print(f'\nEvaluating Model {name}...')

        path = f"saved_models/sasrec_{name}.pth"
        checkpoint = torch.load(path, map_location=device)
        config = SASRecConfig(**checkpoint['config'])

        model = SASRec(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_dataset = SasEvalDataset(test_data, config.max_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        recalls, ndcgs = evaluate(model, test_loader, num_items, device)

        print(f"{name} -> Recall@10: {recalls[10]:.4f}, NDCG@10: {ndcgs[10]:.4f}")
        print(f"{name} -> Recall@20: {recalls[20]:.4f}, NDCG@20: {ndcgs[20]:.4f}")