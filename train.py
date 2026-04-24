import json
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from SASRec import SASRec, get_experiment_config
from evaluate import evaluate, SasEvalDataset


class SASRecDataset(Dataset):
    def __init__(self, data, max_seq_len, num_items):
        self.data = data
        self.max_seq_len = max_seq_len
        self.num_items = num_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        seq = sample["input"]
        target = sample["target"]

        seq = seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(seq)
        seq = [0] * pad_len + seq

        pos = seq[1:] + [target]

        neg = []
        for p in pos:
            if p == 0:
                neg.append(0)
            else:
                n = random.randint(1, self.num_items)
                while n == p:
                    n = random.randint(1, self.num_items)
                neg.append(n)

        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def train_one_version(model, train_loader, val_loader, num_items, device, epochs=10, lr=0.001, patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_ndcg = -1
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for seq, pos, neg in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            loss = model.calculate_loss(seq, pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_recalls, val_ndcgs = evaluate(
            model, val_loader, num_items, device, k_list=[10]
        )
        ndcg_10 = val_ndcgs[10]
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f} - Val NDCG@10: {ndcg_10:.4f}")

        if ndcg_10 > best_ndcg:
            best_ndcg = ndcg_10
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            print('Validation NDCG improved.')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print('Early stopping')
            break

    return best_model_state



def get_num_items(train_data):
    max_item = 0
    for d in train_data:
        max_item = max(max_item, max(d["input"] + [d["target"]]))
    return max_item


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with open("train.json", "r") as f:
        train_data = json.load(f)
    with open("val.json", "r") as f:
        val_data = json.load(f)


    num_items = get_num_items(train_data)
    print("num_items =", num_items)

    os.makedirs("saved_models", exist_ok=True)

    epochs = 10
    batch_size = 128

    for version in ["A", "B", "C"]:
        print("\n" + "=" * 30)
        print(f"TRAINING VERSION {version}")
        print("=" * 30)

        config = get_experiment_config(version, num_items)

        train_dataset = SASRecDataset(
            data=train_data,
            max_seq_len=config.max_seq_len,
            num_items=num_items
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_dataset = SasEvalDataset(
            data=val_data,
            max_seq_len=config.max_seq_len
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False
        )


        model = SASRec(config).to(device)

        train_one_version(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_items=num_items,
            device=device,
            epochs=epochs,
            lr=0.001
        )

        save_path = f"saved_models/sasrec_{version}.pth"
        torch.save(
            {
                "version": version,
                "config": config.__dict__,
                "model_state_dict": model.state_dict(),
            },
            save_path
        )
        print(f"Saved model to: {save_path}")


if __name__ == "__main__":
    main()