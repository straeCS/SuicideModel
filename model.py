import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


# code from ChatGPT ===================== Define LSTM model =====================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # initial layers for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM layers
        out, _ = self.lstm(x, (h0, c0))

        # MLP output layer
        out = self.fc(out[:, -1, :])

        # Sigmoid acvitation to make sure the output is < 1
        out = nn.Sigmoid()(out)
        return out
    




# code from ChatGPT ===================== Define Transformer model =====================

# positional encoding block, source code by the orignial paper
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, nhead=4, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward=hidden_dim), num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.001
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # MLP embedding layer
        src = src.permute(1, 0, 2)
        src = self.embedding(src)

        # Transformer layers
        output = self.transformer_encoder(src)

        # MLP output layer
        output = self.fc(output[-1, :, :])

        # Sigmoid acvitation to make sure the output is < 1
        output = nn.Sigmoid()(output)
        return output






# ===================== Define LinearRegression model ===================== 
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim, timestep):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(input_dim*timestep, output_dim)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.001
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # MLP output layer
        out = self.fc(x[:,:,:].reshape(len(x), -1))

        # Sigmoid acvitation to make sure the output is < 1
        out = nn.Sigmoid()(out)
        return out