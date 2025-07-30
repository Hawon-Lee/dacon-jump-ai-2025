import torch
import torch.nn as nn
from torch_geometric.nn.models import SchNet, AttentiveFP, DimeNetPlusPlus
from torch_geometric.datasets import QM9


class HawonNet(nn.Module):
    def __init__(self, in_features=768, out_features=1, **atten_params):
        super(HawonNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, out_features),
        )

        self.dimenet = DimeNetPlusPlus(
            hidden_channels=128,
            out_channels=1,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6
        )

    def forward(self, x_in):
        # unpack batch
        # mol_emb = x_in.chemberta_mol
        # cls_emb = x_in.chemberta_cls
        # x = x_in.x # graph feature
        # edge_index = x_in.edge_index
        # edge_attr = x_in.edge_attr.unsqueeze(-1) # bond type
        z = x_in.z  # atomic number
        pos = (
            x_in.pos
        )  # atom position (n_total_atoms, n_confs, 3)     n_confs -> 3D 구조 conformer의 갯수 (default=5) / 3 -> x, y, z 위치
        pos = pos[:, 0, :]
        # desc_2d = x_in.desc_2d
        # desc_3d = x_in.desc_3d
        # ecfp = x_in.ecfp
        batch = x_in.batch

        x_out = self.dimenet(z, pos, batch)

        return x_out
