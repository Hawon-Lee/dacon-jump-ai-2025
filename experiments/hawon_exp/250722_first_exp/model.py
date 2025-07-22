import torch
import torch.nn as nn


class HawonNet(nn.Module):
    def __init__(self, in_features=384, out_features=1):
        super(HawonNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
            nn.Linear(in_features//2, out_features)
        )
        

    def forward(self, x_in):
        # unpack batch
        mol_emb = x_in.chemberta_mol
        cls_emb = x_in.chemberta_cls
        x = x_in.x # graph feature
        edge_index = x_in.edge_index
        edge_attr = x_in.edge_attr # bond type
        z = x_in.z # atomic number
        pos = x_in.pos # atom position (n_total_atoms, n_confs, 3)     n_confs -> 3D 구조 conformer의 갯수 (default=5) / 3 -> x, y, z 위치
        desc_2d = x_in.desc_2d
        desc_3d = x_in.desc_3d
        ecfp = x_in.ecfp
        
        return self.layer(mol_emb)
