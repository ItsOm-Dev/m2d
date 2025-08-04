import torch
import torch.nn as nn
import torch.nn.functional as F

class OCRModel(nn.Module):
    def __init__(self, num_classes, num_heads=8, num_layers=4, dim_feedforward=2048):
        super(OCRModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 64, 64, 634]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 128, 32, 317]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Dropout(0.2),
        )

        self.embedding_dim = 512 * 32  # From CNN: [B, 512, 32, W]
        self.pos_encoder = PositionalEncoding(self.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # [B, 512, 32, W]
        B, C, H, W = x.shape

        x = x.permute(0, 3, 1, 2)        # [B, W, C, H]
        x = x.contiguous().view(B, W, C * H)  # [B, W, 512*32]
        x = x.permute(1, 0, 2)           # [W, B, C*H]

        x = self.pos_encoder(x)         # Add position encoding
        x = self.transformer(x)         # [W, B, D]
        x = self.fc(x)                  # [W, B, num_classes]

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [T, B, D]
        x = x + self.pe[:x.size(0)]
        return x
