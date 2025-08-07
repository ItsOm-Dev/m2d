import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return x


class OCRModel(nn.Module):
    def __init__(self, vocab_size, num_heads=4, num_layers=2, dim_feedforward=512, max_len=150):
        super(OCRModel, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = 512

        # 1. CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),    # [B, 64, 128, 512]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),                          # [B, 64, 64, 256]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [B, 128, 64, 256]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),                          # [B, 128, 32, 128]

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # [B, 256, 32, 128]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),                          # [B, 256, 16, 64]

            nn.Dropout(0.1)
        )

        # 2. Flatten CNN features
        self.embedding_dim = 256 * 16  # = 4096
        self.input_proj = nn.Linear(self.embedding_dim, self.d_model)

        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(self.d_model)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=0.1,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Token Embedding for target sequence (Decoder input)
        self.token_embedding = nn.Embedding(vocab_size, self.d_model)

        # 6. Positional Encoding for decoder
        self.pos_decoder = PositionalEncoding(self.d_model)

        # 7. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=0.1,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 8. Final projection to vocabulary
        self.generator = nn.Linear(self.d_model, vocab_size)

    def forward(self, images, tgt_seq, tgt_mask=None):
        """
        images: [B, 1, 128, 512]
        tgt_seq: [B, T] => token ids including <sos> (for teacher forcing)
        """

        # CNN Encoder
        x = self.cnn(images)                     # [B, 256, 16, 64]
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2)                # [B, 64, 256, 16]
        x = x.reshape(B, W, C * H)               # [B, 64, 4096]
        x = self.input_proj(x)                   # [B, 64, 512]
        x = self.pos_encoder(x)                  # +positional encoding

        # Transformer Encoder
        memory = self.encoder(x)                 # [B, 64, 512]

        # Prepare target sequence
        tgt_emb = self.token_embedding(tgt_seq)  # [B, T, 512]
        tgt_emb = self.pos_decoder(tgt_emb)      # +positional encoding

        # Transformer Decoder
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)  # [B, T, 512]

        # Project to vocabulary
        output = self.generator(output)          # [B, T, vocab_size]

        return output
