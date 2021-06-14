import torch
import torch.nn as nn

from models.constants import ACTIVATIONS
from typing import Dict, Optional


class DependencyParserV1(nn.Module):
    @classmethod
    def from_params(cls, words_vocab_size: int, poses_vocab_size: int, model_params: Dict):
        words_embedding_dim = int(model_params.get("words_embedding_dim", 128))
        poses_embedding_dim = int(model_params.get("poses_embedding_dim", 32))
        lstm_num_layers = int(model_params.get("lstm_num_layers", 2))
        lstm_hidden_dim = int(model_params.get("lstm_hidden_dim", 128))
        lstm_dropout = float(model_params.get("lstm_dropout", 0.0))
        linear_output_dim = int(model_params.get("linear_output_dim", 128))
        activation_type = model_params.get("activation_type", "tanh")

        return cls(
            words_vocab_size=words_vocab_size,
            poses_vocab_size=poses_vocab_size,
            words_embedding_dim=words_embedding_dim,
            poses_embedding_dim=poses_embedding_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_dropout=lstm_dropout,
            linear_output_dim=linear_output_dim,
            activation_type=activation_type,
        )

    def __init__(
        self,
        words_vocab_size: int,
        poses_vocab_size: int,
        words_embedding_dim: Optional[int] = 128,
        poses_embedding_dim: Optional[int] = 32,
        lstm_num_layers: Optional[int] = 2,
        lstm_hidden_dim: Optional[int] = 128,
        lstm_dropout: Optional[float] = 0.0,
        linear_output_dim: Optional[int] = 128,
        activation_type: Optional[str] = "tanh",
        activation_params: Optional[Dict] = dict(),
    ):
        super(DependencyParserV1, self).__init__()

        if not 0 <= lstm_dropout <= 1:
            raise ValueError(lstm_dropout)

        self.words_embedding = nn.Embedding(num_embeddings=words_vocab_size, embedding_dim=words_embedding_dim)
        self.poses_embedding = nn.Embedding(num_embeddings=poses_vocab_size, embedding_dim=poses_embedding_dim)
        self.LSTM = nn.LSTM(
            input_size=(words_embedding_dim + poses_embedding_dim),
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=True,
            dropout=lstm_dropout,
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
