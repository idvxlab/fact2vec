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
        init_x =list(map(lambda xx: self.bert.encode(xx), x))
        bert_x = torch.as_tensor(init_x)
        fc_out = self.fc(bert_x)
        return fc_out

