import torch
import torch.nn as nn

from models.constants import ACTIVATIONS
from typing import Dict, Optional, Tuple


class DependencyParserV1(nn.Module):
    def __init__(
        self,
        words_vocab_size: int,
        poses_vocab_size: int,
        words_embedding_dim: Optional[int] = 128,
        poses_embedding_dim: Optional[int] = 32,
        lstm_num_layers: Optional[int] = 2,
        lstm_hidden_dim: Optional[int] = 128,
        lstm_droput: Optional[float] = 0.0,
        linear_output_dim: Optional[int] = 128,
        activation_type: Optional[str] = "tanh",
        activation_params: Optional[Dict] = dict(),
    ):
        super(DependencyParserV1, self).__init__()

        self.words_embedding = nn.Embedding(num_embeddings=words_vocab_size, embedding_dim=words_embedding_dim)
        self.poses_embedding = nn.Embedding(num_embeddings=poses_vocab_size, embedding_dim=poses_embedding_dim)
        self.LSTM = nn.LSTM(
            input_size=(words_embedding_dim + poses_embedding_dim),
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=True,
            dropout=lstm_droput,
            batch_first=True,
        )
        linear_proj_input_dim = 2 * lstm_hidden_dim
        self.h_linear = nn.Linear(linear_proj_input_dim, linear_output_dim)
        self.m_linear = nn.Linear(linear_proj_input_dim, linear_output_dim)
        self.act = ACTIVATIONS[activation_type](**activation_params)
        self.linear = nn.Linear(linear_output_dim, 1)

    def forward(self, words_embedding_indices: torch.Tensor, poses_embedding_indices: torch.Tensor):

        # Embedding
        words_embedding = self.words_embedding(words_embedding_indices)
        poses_embedding = self.poses_embedding(poses_embedding_indices)
        embedding = torch.cat(tensors=(words_embedding, poses_embedding), dim=2)

        # Feed to LSTM
        LSTM_output, _ = self.LSTM(input=embedding)

        # Linear projection
        h_out = self.h_linear(LSTM_output)
        m_out = self.m_linear(LSTM_output)

        # Compute scores
        scores = torch.unsqueeze(h_out, 2) + torch.unsqueeze(m_out, 1)
        scores = self.act(scores)
        scores = self.linear(scores)
        scores = scores.squeeze(-1)

        return scores
