import torch
import torch.nn as nn
from torch_geometric.nn.models import SchNet, AttentiveFP, DimeNetPlusPlus
from torch_geometric.datasets import QM9
import sys

sys.path.append("/home/tech/Hawon/Develop/Interaction_free/MAIN_REFAC/new/SCRIPTS/")
from model.model_encoders import MoleculeEncoder_CustomEGNN


class HawonNet(nn.Module):
    def __init__(self, in_features=768, out_features=1, **atten_params):
        super(HawonNet, self).__init__()

        self.egnn = MoleculeEncoder_CustomEGNN(output_dim=1, num_interactions=6)
        self.egnn.reset_parameters()
        
        self.afp = AttentiveFP(
            in_channels=54,
            hidden_channels=128,
            out_channels=1,
            edge_dim=1,
            num_layers=3,
            num_timesteps=3,
            dropout=0.1
        )
        self.afp.reset_parameters()

        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_in):
        ############ unpack batch ############
        # mol_emb = x_in.chemberta_mol
        # cls_emb = x_in.chemberta_cls
        x = x_in.x # graph feature
        edge_index = x_in.edge_index
        edge_attr = x_in.edge_attr.unsqueeze(-1) # bond type
        z = x_in.z  # atomic number
        pos = (
            x_in.pos
        )  # atom position (n_total_atoms, n_confs, 3)     n_confs -> 3D 구조 conformer의 갯수 (default=5) / 3 -> x, y, z 위치
        pos = pos[:, 0, :]
        # desc_2d = x_in.desc_2d
        # desc_3d = x_in.desc_3d
        # ecfp = x_in.ecfp
        batch = x_in.batch
        label = x_in.label
        #####################################
        
        x_1 = self.egnn(z, pos, batch)
        x_2 = self.afp(x, edge_index, edge_attr, batch)
        
        # x = torch.concat([x_1, x_2], dim=1)
        # x_out = self.mlp(x)
        
        return x_1 + x_2
