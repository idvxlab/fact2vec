import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer

class FactEmbedder(nn.Module):
    def __init__(self):
        super(FactEmbedder, self).__init__()
        self.bert = SentenceTransformer('bert-base-nli-mean-tokens')

        self.fc = nn.Sequential(
            nn.Linear(768, 768, bias=True),
            nn.Linear(768, 768, bias=True),
        )

    def forward(self, x):
        init_x = torch.as_tensor(self.bert.encode(x))
        fc_out = self.fc(init_x)
        return fc_out

