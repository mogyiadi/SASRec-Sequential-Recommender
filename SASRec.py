import math
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SASRecConfig:
    num_items: int
    max_seq_len: int = 50
    hidden_size: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout_rate: float = 0.2
    initializer_range: float = 0.02


def get_experiment_config(version: str, num_items: int) -> SASRecConfig:

    versions = {
        # version A: baseline
        "A": {
            "max_seq_len": 50,
            "hidden_size": 64,
            "num_blocks": 2,
            "num_heads": 2,
            "dropout_rate": 0.2,
        },

        # version B: deeper model
        "B": {
            "max_seq_len": 50,
            "hidden_size": 64,
            "num_blocks": 4,
            "num_heads": 2,
            "dropout_rate": 0.2,
        },

        # version C: wider model
        "C": {
            "max_seq_len": 50,
            "hidden_size": 128,
            "num_blocks": 2,
            "num_heads": 4,
            "dropout_rate": 0.2,
        },
    }

    cfg = versions[version]

    if cfg["hidden_size"] % cfg["num_heads"] != 0:
        raise ValueError(
            f"hidden_size ({cfg['hidden_size']}) must be divisible by num_heads ({cfg['num_heads']})"
        )

    return SASRecConfig(
        num_items=num_items,
        max_seq_len=cfg["max_seq_len"],
        hidden_size=cfg["hidden_size"],
        num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"],
        dropout_rate=cfg["dropout_rate"],
    )


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden]
        out = x.transpose(1, 2)              # [batch, hidden, seq_len]
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        out = out.transpose(1, 2)            # [batch, seq_len, hidden]
        return out


class SASRecBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()

        self.attn_layer_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout_rate)

        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = PointWiseFeedForward(hidden_size, dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        causal_mask: torch.Tensor
    ) -> torch.Tensor:
        # x: [batch, seq_len, hidden]
        # padding_mask: [batch, seq_len] -> True where padding
        # causal_mask: [seq_len, seq_len] -> True where future positions are masked

        residual = x
        x_norm = self.attn_layer_norm(x)

        attn_output, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = residual + self.attn_dropout(attn_output)

        residual = x
        x_norm = self.ffn_layer_norm(x)
        x = residual + self.ffn(x_norm)

        return x


class SASRec(nn.Module):
    def __init__(self, config: SASRecConfig):
        super().__init__()
        self.config = config
        self.num_items = config.num_items
        self.max_seq_len = config.max_seq_len
        self.hidden_size = config.hidden_size
        self.num_blocks = config.num_blocks
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate

        self.item_embedding = nn.Embedding(
            num_embeddings=self.num_items + 1,
            embedding_dim=self.hidden_size,
            padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=self.max_seq_len,
            embedding_dim=self.hidden_size
        )

        self.embedding_dropout = nn.Dropout(self.dropout_rate)

        self.blocks = nn.ModuleList([
            SASRecBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )
            for _ in range(self.num_blocks)
        ])

        self.final_layer_norm = nn.LayerNorm(self.hidden_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _get_position_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # True means masked
        return torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1
        )

    def log2feats(self, input_seq: torch.Tensor) -> torch.Tensor:
       
        device = input_seq.device
        batch_size, seq_len = input_seq.shape

        if seq_len > self.max_seq_len:
            input_seq = input_seq[:, -self.max_seq_len:]
            seq_len = self.max_seq_len

        positions = self._get_position_ids(seq_len, device).expand(batch_size, -1)

        seq_emb = self.item_embedding(input_seq) * math.sqrt(self.hidden_size)
        pos_emb = self.position_embedding(positions)

        x = seq_emb + pos_emb
        x = self.embedding_dropout(x)

        padding_mask = (input_seq == 0)                 # [batch, seq_len]
        causal_mask = self._get_causal_mask(seq_len, device)   # [seq_len, seq_len]

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, causal_mask=causal_mask)

        x = self.final_layer_norm(x)

        # zero out padded positions
        x = x * (~padding_mask).unsqueeze(-1)

        return x

    def forward(
        self,
        input_seq: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor
    ):
    
        sequence_output = self.log2feats(input_seq)  # [batch, seq_len, hidden]

        pos_emb = self.item_embedding(positive_items)   # [batch, seq_len, hidden]
        neg_emb = self.item_embedding(negative_items)   # [batch, seq_len, hidden]

        positive_logits = torch.sum(sequence_output * pos_emb, dim=-1)  # [batch, seq_len]
        negative_logits = torch.sum(sequence_output * neg_emb, dim=-1)  # [batch, seq_len]

        return sequence_output, positive_logits, negative_logits

    def calculate_loss(
        self,
        input_seq: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor
    ) -> torch.Tensor:
        
        _, pos_logits, neg_logits = self.forward(input_seq, positive_items, negative_items)

        valid_mask = (positive_items != 0).float()

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits),
            reduction="none"
        )

        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits,
            torch.zeros_like(neg_logits),
            reduction="none"
        )

        loss = (pos_loss + neg_loss) * valid_mask
        loss = loss.sum() / torch.clamp(valid_mask.sum(), min=1.0)
        return loss

    def get_last_hidden_state(self, input_seq: torch.Tensor) -> torch.Tensor:
        
        seq_output = self.log2feats(input_seq)  # [batch, seq_len, hidden]

        non_pad_idx = (input_seq != 0).sum(dim=1) - 1
        non_pad_idx = torch.clamp(non_pad_idx, min=0)

        batch_idx = torch.arange(input_seq.size(0), device=input_seq.device)
        last_hidden = seq_output[batch_idx, non_pad_idx]  # [batch, hidden]

        return last_hidden

    def predict(
        self,
        input_seq: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        last_hidden = self.get_last_hidden_state(input_seq)  # [batch, hidden]

        if candidate_items is None:
            all_item_emb = self.item_embedding.weight[1:]  # skip padding index 0
            scores = torch.matmul(last_hidden, all_item_emb.t())  # [batch, num_items]
            return scores

        candidate_emb = self.item_embedding(candidate_items)  # [batch, num_candidates, hidden]
        scores = torch.sum(last_hidden.unsqueeze(1) * candidate_emb, dim=-1)
        return scores


def run_demo(version: str, num_items: int, batch_size: int = 4, seq_len: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_experiment_config(version, num_items)
    model = SASRec(config).to(device)

    print(f"\n==============================")
    print(f" VERSION {version}")
    print(f"==============================")
    print(config)

    # Dummy data
    input_seq = torch.randint(1, num_items + 1, (batch_size, seq_len), device=device)
    positive_items = torch.randint(1, num_items + 1, (batch_size, seq_len), device=device)
    negative_items = torch.randint(1, num_items + 1, (batch_size, seq_len), device=device)

    # Add padding to test masking
    input_seq[0, :2] = 0
    positive_items[0, :2] = 0
    negative_items[0, :2] = 0

    _, pos_logits, neg_logits = model(input_seq, positive_items, negative_items)

    loss = model.calculate_loss(input_seq, positive_items, negative_items)

    print("Loss:", loss.item())

    scores = model.predict(input_seq)
    print("Output shape:", scores.shape)


def main():
    num_items = 3533   

    batch_size = 4     
    seq_len = 10       

    for version in ["A", "B", "C"]:
        run_demo(
            version=version,
            num_items=num_items,
            batch_size=batch_size,
            seq_len=seq_len
        )


if __name__ == "__main__":
    main()