import torch
import torch.nn as nn
from torch_geometric.nn.models import SchNet, AttentiveFP, DimeNetPlusPlus
from torch_geometric.datasets import QM9
import random
import sys
sys.path.append("/home/tech/Hawon/Develop/Interaction_free/MAIN_REFAC/new/SCRIPTS/")
from model.model_encoders import MoleculeEncoder_CustomEGNN

class HawonNet(nn.Module):
    def __init__(self, in_features=768, out_features=1, **atten_params):
        super(HawonNet, self).__init__()
        
        self.egnn = MoleculeEncoder_CustomEGNN(
            output_dim=1,
            num_interactions=4
        )

    def forward(self, x_in):
        ############ unpack batch ############
        # mol_emb = x_in.chemberta_mol
        # cls_emb = x_in.chemberta_cls
        # x = x_in.x # graph feature
        # edge_index = x_in.edge_index
        # edge_attr = x_in.edge_attr.unsqueeze(-1) # bond type
        z = x_in.z  # atomic number
        pos = (x_in.pos)  # atom position (n_total_atoms, n_confs, 3)     n_confs -> 3D 구조 conformer의 갯수 (default=5) / 3 -> x, y, z 위치
        conf_idx = random.randint(0, 4)  # 1~4 중 랜덤 선택 (0 제외)
        pos = pos[:, conf_idx, :]
        # desc_2d = x_in.desc_2d
        # desc_3d = x_in.desc_3d
        # ecfp = x_in.ecfp
        batch = x_in.batch
        #####################################

        x_out = self.egnn(z, pos, batch)
        
        return x_out
