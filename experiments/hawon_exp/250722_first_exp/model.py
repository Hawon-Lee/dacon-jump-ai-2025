import torch
import torch.nn as nn


class HawonNet(nn.Module):
    def __init__(self, in_feature=384, out_feature=1):
        super(HawonNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_feature, in_feature//2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
            nn.Linear(in_feature//2, out_feature)
        )
        

    def forward(self, x_in):
        # unpack batch
        chemberta_mol = x_in.chemberta_mol
        chemberta_cls = x_in.chemberta_cls
        x = x_in.x
        # ...
        
        return self.layer(chemberta_mol)