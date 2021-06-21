import math
import torch
import torch.nn as nn

from models.constants import ACTIVATIONS
from torch.autograd import Variable
from typing import Dict, Optional
from utils.torch import to_device


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 200):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + to_device(Variable(self.pe[:, :seq_len], requires_grad=False))
        return x


class DependencyParserV2(nn.Module):
    @classmethod
    def from_params(cls, words_vocab_size: int, poses_vocab_size: int, model_params: Dict):
        words_embedding_dim = int(model_params.get("words_embedding_dim", 128))
        poses_embedding_dim = int(model_params.get("poses_embedding_dim", 32))
        transformer_nhead = int(model_params.get("transformer_nhead", 4))
        transformer_hidden_dim = int(model_params.get("transformer_hidden_dim", 128))
        transformer_dropout = float(model_params.get("transformer_dropout", 0.1))
        transformer_num_layers = int(model_params.get("transformer_num_layers", 4))
        activation_type = model_params.get("activation_type", "tanh")

        return cls(
            words_vocab_size=words_vocab_size,
            poses_vocab_size=poses_vocab_size,
            words_embedding_dim=words_embedding_dim,
            poses_embedding_dim=poses_embedding_dim,
            transformer_nhead=transformer_nhead,
            transformer_hidden_dim=transformer_hidden_dim,
            transformer_dropout=transformer_dropout,
            transformer_num_layers=transformer_num_layers,
            activation_type=activation_type,
        )

    def __init__(
        self,
        words_vocab_size: int,
        poses_vocab_size: int,
        words_embedding_dim: Optional[int] = 128,
        poses_embedding_dim: Optional[int] = 32,
        transformer_nhead: Optional[int] = 4,
        transformer_hidden_dim: Optional[int] = 128,
        transformer_dropout: Optional[float] = 0.0,
        transformer_num_layers: Optional[int] = 4,
        linear_output_dim: Optional[int] = 128,
        activation_type: Optional[str] = "tanh",
        activation_params: Optional[Dict] = dict(),
    ):

        super(DependencyParserV2, self).__init__()

        self.words_embedding = nn.Embedding(num_embeddings=words_vocab_size, embedding_dim=words_embedding_dim)
        self.poses_embedding = nn.Embedding(num_embeddings=poses_vocab_size, embedding_dim=poses_embedding_dim)

        self.src_mask = None
        self.d_model = words_embedding_dim + poses_embedding_dim
        self.pos_encoder = PositionalEncoder(d_model=self.d_model)

        self.dim_feedforward = 2 * transformer_hidden_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=transformer_nhead, dim_feedforward=self.dim_feedforward, dropout=transformer_dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=transformer_num_layers)
        self.h_linear = nn.Linear(self.d_model, linear_output_dim)
        self.m_linear = nn.Linear(self.d_model, linear_output_dim)
        self.act = ACTIVATIONS[activation_type](**activation_params)
        self.linear = nn.Linear(linear_output_dim, 1)

    def _generate_square_subsequent_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, words_embedding_indices: torch.Tensor, poses_embedding_indices: torch.Tensor):

        # Embedding
        words_embedding = self.words_embedding(words_embedding_indices)
        poses_embedding = self.poses_embedding(poses_embedding_indices)
        embedding = torch.cat(tensors=(words_embedding, poses_embedding), dim=2)

        if self.src_mask is None or self.src_mask.size(0) != len(embedding):
            mask = to_device(self._generate_square_subsequent_mask(length=len(embedding)))
            self.src_mask = mask

        # Feed to Transformer
        src = embedding * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        trasformer_output = self.transformer_encoder(src=src, mask=self.src_mask)

        # Linear projection
        h_out = self.h_linear(trasformer_output)
        m_out = self.m_linear(trasformer_output)

        # Compute scores
        scores = torch.unsqueeze(h_out, 2) + torch.unsqueeze(m_out, 1)
        scores = self.act(scores)
        scores = self.linear(scores)
        scores = scores.squeeze(-1)

        return scores
