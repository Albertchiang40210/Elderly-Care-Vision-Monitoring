import torch
import torch.nn as nn

class ActionTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=34, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        """
        時間序列骨架特徵的 Transformer 分類器
        :param input_dim: 每個時間點的特徵維度 (17個關節 * 2(x,y) = 34)
        :param d_model: Transformer 隱藏層維度
        :param nhead: 多頭注意力機制的頭數
        :param num_layers: Encoder 層數
        """
        super(ActionTransformer, self).__init__()
        
        # 1. 座標特徵映射到高維
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Transformer Encoder 核心
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True # 輸入維度設定為 (Batch, Sequence, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 分類頭 (使用 Sequence 中最後一個輸出來進行分類，或是取平均)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        # (例如: [32, 30, 34])
        
        x = self.embedding(x) # [batch_size, sequence_length, d_model]
        
        # 經過 Transformer Encoder
        out = self.transformer_encoder(x) # [batch_size, sequence_length, d_model]
        
        # 取時間序列的最後一個狀態作為分類依據
        # out[:, -1, :] 取出最後一個時間步的特徵
        last_hidden_state = out[:, -1, :] # [batch_size, d_model]
        
        # 輸出類別預測
        logits = self.classifier(last_hidden_state)
        return logits
