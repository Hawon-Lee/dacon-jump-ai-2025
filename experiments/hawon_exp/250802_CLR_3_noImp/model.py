import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import SchNet, AttentiveFP, DimeNetPlusPlus
from torch_geometric.datasets import QM9
import sys

sys.path.append("/home/tech/Hawon/Develop/Interaction_free/MAIN_REFAC/new/SCRIPTS/")
from model.model_encoders import MoleculeEncoder_CustomEGNN


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, similarity_threshold=0.5):
        super().__init__()
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold

    def forward(self, embeddings, labels):
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask based on pIC50 similarity
        label_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
        positive_mask = (label_diff < self.similarity_threshold).float()
        positive_mask.fill_diagonal_(0)

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        pos_sum = torch.sum(exp_sim * positive_mask, dim=1)
        all_sum = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim)

        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        return loss.mean()


class HawonNet(nn.Module):
    def __init__(
        self,
        in_features=768,
        out_features=1,
        use_contrastive=True,
        contrastive_weight=0.1,
        **atten_params,
    ):
        super(HawonNet, self).__init__()

        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight

        self.egnn = MoleculeEncoder_CustomEGNN(
            hidden_channels=128,
            output_dim=64,
            num_interactions=6,
            dropout=0.1
        )
        self.egnn.reset_parameters()

        self.afp = AttentiveFP(
            in_channels=54,
            hidden_channels=128,
            out_channels=64,
            edge_dim=1,
            num_layers=3,
            num_timesteps=3,
            dropout=0.1,
        )
        self.afp.reset_parameters()

        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

        # Contrastive loss 추가
        if self.use_contrastive:
            self.contrastive_loss_fn = ContrastiveLoss(
                temperature=0.07, similarity_threshold=0.5
            )

    def forward(self, x_in):
        ############ unpack batch ############
        x = x_in.x
        edge_index = x_in.edge_index
        edge_attr = x_in.edge_attr.unsqueeze(-1)
        z = x_in.z
        pos = x_in.pos[:, 0, :]
        batch = x_in.batch
        label = x_in.get("label", None)
        #####################################

        x_1 = self.egnn(z, pos, batch)
        x_2 = self.afp(x, edge_index, edge_attr, batch)
        x_cat = torch.cat([x_1, x_2], dim=1)
        # Contrastive loss 계산 (training 시에만)
        total_loss = 0
        if (
            self.use_contrastive and self.training and x.size(0) > 1
        ):  # batch size > 1 일 때만
            contrastive_loss = self.contrastive_loss_fn(x_cat, label)
            total_loss = self.contrastive_weight * contrastive_loss

        # Main prediction
        main_output = self.mlp(x_cat)

        if self.training:
            return main_output, total_loss
        else:
            return main_output
