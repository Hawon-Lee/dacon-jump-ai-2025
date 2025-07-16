from typing import Literal
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D, rdFingerprintGenerator
import torch


def get_descriptors_2d(
    mol: Chem.rdchem.Mol,
    descriptors_list: list[str] = [
        "MolWt",
        "MolLogP",
        "TPSA",
        "NumHAcceptors",
        "NumHDonors",
        "NumRotatableBonds",
    ],
) -> torch.Tensor:

    desc_values = []

    computed_desc = Descriptors.CalcMolDescriptors(mol)

    for desc_type in descriptors_list:
        desc_value = computed_desc.get(desc_type)
        if desc_value is None:
            print(
                f"Warning: Descriptor type {desc_type} was not found. The corresponding value is filled with 0."
            )
            desc_value = 0.0

        desc_values.append(desc_value)

    return torch.tensor(desc_values, dtype=torch.float32)


def get_descriptors_3d(
    mol: Chem.rdchem.Mol, method: Literal["avg", "sum", "concat"] = "avg"
) -> torch.Tensor:

    if method not in ["avg", "sum", "concat"]:
        raise ValueError("method must be avg, sum, or concat")

    pooled_desc_values = []
    for i in range(len(mol.GetConformers())):
        computed_desc = Descriptors3D.CalcMolDescriptors3D(mol, confId=i)
        descriptors_values = [v for k, v in computed_desc.items()]
        pooled_desc_values.append(descriptors_values)

    pooled_desc_values = torch.tensor(pooled_desc_values, dtype=torch.float32)
    if method == "avg":
        return pooled_desc_values.mean(dim=0)
    elif method == "sum":
        return pooled_desc_values.sum(dim=0)
    elif method == "concat":
        return torch.flatten(pooled_desc_values)


def get_ecfp(mol: Chem.rdchem.Mol, radius=2, fpSize=2048) -> torch.Tensor:
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fpSize
    )
    
    fp = fp_generator.GetFingerprintAsNumPy(mol)
    return torch.from_numpy(fp)

# def get_node_feature(mol):

# def get_edge_indice(mol):