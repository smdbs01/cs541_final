import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder with MHA and Positional Encoding
    """

    def __init__(self, input_dim, emb_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(input_dim, emb_dim)

    def positional_encoding(self, seq_len, emb_dim):
        pe = torch.zeros(seq_len, emb_dim)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(np.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.fc(x)
        x = x + self.positional_encoding(seq_len, self.emb_dim).to(x.device)
        x = self.dropout_layer(x)
        x = self.layer_norm(x)
        x, _ = self.mha(x, x, x)
        x = self.layer_norm(x)
        return x.mean(dim=1)


class CNNLSTM(nn.Module):
    # Input is (batch_size, channels, num_frames, width, height)
    def __init__(self, input_channels, hidden_size, num_layers, num_classes):
        super(CNNLSTM, self).__init__()

        self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # type: ignore

        # self.lstm = nn.LSTM(
        #     input_size=512,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     bidirectional=False,
        # )

        self.transformer = TransformerEncoder(
            input_dim=512,
            emb_dim=hidden_size,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        batch_size, C, T, H, W = x.size()

        x = x.permute(0, 2, 1, 3, 4)
        dx = x[:, 1:, :, :, :] - x[:, :-1, :, :, :]

        x = dx
        x = x.reshape(-1, C, H, W)

        cnn_features = self.resnet(x)
        cnn_features = cnn_features.view(batch_size, T - 1, -1)
        # (batch_size, T, 512)

        # lstm_out, _ = self.lstm(cnn_features)
        transformer_out = self.transformer(cnn_features)
        # (batch_size, 512)
        out = self.fc(transformer_out)
        return out
