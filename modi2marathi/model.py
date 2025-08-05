import torch
import torch.nn as nn

class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()

        self.cnn = nn.Sequential(
            # Input: [B, 1, 128, 1268]
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),   # → [B, 64, 128, 1268]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # → [B, 64, 64, 634]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # → [B, 128, 64, 634]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # → [B, 128, 32, 317]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# → [B, 256, 32, 317]
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),# → [B, 256, 32, 317]
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),# → [B, 512, 32, 317]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),# → [B, 512, 32, 317]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Dropout(0.2),  # optional regularization
        )

        # RNN input: channels × height = 512 × 32
        self.rnn_input_size = 512 * 32
        self.hidden_size = 256

        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.hidden_size,
            num_layers=3,  # was 2 before
            bidirectional=True,
            batch_first=False
        )

        self.fc = nn.Linear(self.hidden_size * 2, num_classes)

    def forward(self, x):
        # CNN output shape: [B, 512, 32, 317]
        x = self.cnn(x)
        B, C, H, W = x.size()

        # Reshape to [T, B, F]
        x = x.permute(0, 3, 1, 2)            # [B, W=317, C=512, H=32]
        x = x.contiguous().view(B, W, C * H) # [B, 317, 512×32]
        x = x.permute(1, 0, 2)               # [T=317, B, F=16384]

        # LSTM → FC
        x, _ = self.lstm(x)                  # [T=317, B, 512]
        x = self.fc(x)                       # [T=317, B, num_classes]

        return x









########################################OLD WORKING MODEL #################################


# import torch
# import torch.nn as nn

# class OCRModel(nn.Module):
#     def __init__(self, num_classes):
#         super(OCRModel, self).__init__()

#         self.cnn = nn.Sequential(
#             # Input: [B, 1, 128, 1268]
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),   # → [B, 64, 128, 1268]
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),                  # → [B, 64, 64, 634]

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # → [B, 128, 64, 634]
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),                  # → [B, 128, 32, 317]

#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# → [B, 256, 32, 317]
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),# → [B, 256, 32, 317]
#             nn.ReLU(),

#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),# → [B, 512, 32, 317]
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             # ❌ No more downsampling: preserve width and height
#         )

#         self.rnn_input_size = 512 * 32 #before it was 24   # channels × height
#         self.hidden_size = 256

#         self.lstm = nn.LSTM(
#             input_size=self.rnn_input_size,
#             hidden_size=self.hidden_size,
#             num_layers=3,#it was 2 before#if not good make it 3
#             bidirectional=True,
#             batch_first=False
#         )

#         self.fc = nn.Linear(self.hidden_size * 2, num_classes)

#     def forward(self, x):
#         x = self.cnn(x)  # [B, 512, 32, 317]
#         B, C, H, W = x.size()

#         # Reshape for RNN input: (T, B, F)
#         x = x.permute(0, 3, 1, 2)            # [B, W=317, C, H=32]
#         x = x.contiguous().view(B, W, C * H) # [B, W, C×H] = [B, 317, 512×32]
#         x = x.permute(1, 0, 2)  
#         '''
#         FOR DEBUG
#         print(f"expected CNN  [B, 512, 32, 317]")             # [T=317, B, F]
#         print(f"✅ got CNN out: {x.shape}") 
#         print(f"expected LSTM After reshape: [T=317, B, 16384] ")             # [T=317, B, F]
#               # [B, 512, 32, 317]
#         print(f"✅got LSTM in: {x.shape}") 
              
#         # After reshape: [T=317, B, 16384]
#         '''
#         x, _ = self.lstm(x)                  # [T, B, 2×hidden]
#         x = self.fc(x)                       # [T, B, num_classes]

#         return x
