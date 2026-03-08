import os
import esm
import torch
import warnings
import argparse
import pickle as pk
import numpy as np
import torch.nn.functional as F
from utils import *
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
warnings.simplefilter('ignore')


class BasePDB(Dataset):
    def __init__(self, mode='train', fold=-1, root='./data/BCE_633', self_cycle=False, sub_dir=None):
        self.root = root
        self.sub_dir = sub_dir
        assert mode in ['train', 'val', 'test']

        pkl_file = 'train.pkl' if mode in ['train', 'val'] else 'test.pkl'
        with open(f'{self.root}/{pkl_file}', 'rb') as f:
            self.samples = pk.load(f)

        idx = np.load(f'{self.root}/cross-validation.npy')
        cv, inter, ex = 10, len(idx) // 10, len(idx) % 10

        if mode == 'train':
            order = [j for i in range(cv) if i != fold
                     for j in idx[i*inter:(i+1)*inter + ex*(i == cv-1)]]
        elif mode == 'val':
            order = list(idx[fold*inter:(fold+1)*inter + ex*(fold == cv-1)])
        else:
            order = list(range(len(self.samples)))

        self.data = []
        tbar = tqdm(sorted(order))
        for i in tbar:
            tbar.set_postfix(chain=self.samples[i].name)
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root, self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)

    def _base_item(self, idx):
        seq = self.data[idx]
        return seq, torch.cat([seq.feat, seq.dssp], 1)

    def _pack(self, seq, feat):
        return {'feat': feat, 'label': seq.label, 'adj': seq.adj, 'edge': seq.edge}

    def __getitem__(self, idx):
        raise NotImplementedError


# ── Hardcoded external data paths ─────────────────────────────────────────────
_FOLD_SEEK_DIR = "/work/home/maben/project/epitope_prediction/GraphBepi/data/fold_seek_vectors"
_FOLD_SEEK_GANGXU = "/work/home/maben/project/epitope_prediction/GraphBepi/data/fold_seek_vectors_gang_xu"
_ESM_IF_DIR = "/work/home/maben/project/epitope_prediction/GraphBepi/data/BCE_633_fold_seek/esm-if"
_SAPORT_DIR = "/work/home/maben/project/epitope_prediction/SaProt/get_inputs/saport_embeddings"
_STRUCT_BASE = "/work/home/xugang/projects/struct_diff/maben"
_ESM_FEAT_DIR = "/work/home/xugang/projects/struct_diff/maben/feat_esm"
# ──────────────────────────────────────────────────────────────────────────────


class PDB(BasePDB):
    def __getitem__(self, idx):
        seq, feat = self._base_item(idx)
        return self._pack(seq, feat)


class PDB_foldseek(BasePDB):
    def __getitem__(self, idx):
        seq, base_feat = self._base_item(idx)
        vector = torch.load(os.path.join(_FOLD_SEEK_DIR, f"{seq.name}.pt"))
        onehot = F.one_hot(vector.long(), num_classes=21).permute(0, 2, 1).reshape(21*6, -1).t()
        return self._pack(seq, torch.cat([base_feat, onehot], 1))


class PDB_foldseek_local_golbal(BasePDB):
    def __getitem__(self, idx):
        seq, base_feat = self._base_item(idx)
        name = seq.name

        vec_global = torch.load(os.path.join(_FOLD_SEEK_GANGXU, f"{name}.pt")).unsqueeze(0)
        onehot_global = F.one_hot(vec_global.long(), num_classes=21).permute(0, 2, 1).reshape(21, -1).t()

        vec_local = torch.load(os.path.join(_FOLD_SEEK_DIR, f"{name}.pt"))[0].unsqueeze(0)
        onehot_local = F.one_hot(vec_local.long(), num_classes=21).permute(0, 2, 1).reshape(21, -1).t()

        return self._pack(seq, torch.cat([base_feat, onehot_global, onehot_local], 1))


class PDB_foldseek_attn(BasePDB):
    def __getitem__(self, idx):
        seq, base_feat = self._base_item(idx)
        vector = torch.load(os.path.join(_FOLD_SEEK_DIR, f"{seq.name}.pt"))
        onehot = F.one_hot(vector.long(), num_classes=21).permute(0, 2, 1).reshape(-1, 21, 6)
        return {**self._pack(seq, base_feat), 'fold_token': onehot}


class PDB_foldseek_tokens(BasePDB):
    def __getitem__(self, idx):
        seq, base_feat = self._base_item(idx)
        vector = torch.load(os.path.join(_FOLD_SEEK_GANGXU, f"{seq.name}.pt")).unsqueeze(0)
        onehot = F.one_hot(vector.long(), num_classes=21).permute(0, 2, 1).reshape(21, -1).t()
        return self._pack(seq, torch.cat([base_feat, onehot], 1))


class PDB_token_esm(BasePDB):
    def __getitem__(self, idx):
        seq, base_feat = self._base_item(idx)
        vector = torch.Tensor(
            np.load(os.path.join(_ESM_FEAT_DIR, f"{seq.name}.feat_esm.npz"))["f"]
        ).unsqueeze(0).permute(0, 2, 1).reshape(640, -1).t()
        return self._pack(seq, torch.cat([base_feat, vector], 1))


class PDB_esm_if(BasePDB):
    def __getitem__(self, idx):
        seq, base_feat = self._base_item(idx)
        vector = torch.load(os.path.join(_ESM_IF_DIR, f"{seq.name}.pt")).unsqueeze(0)
        vector = vector.permute(0, 2, 1).reshape(512, -1).t()
        return self._pack(seq, torch.cat([base_feat, vector], 1))


class PDB_esm_if_foldseek_tokens(BasePDB):
    def __getitem__(self, idx):
        seq, base_feat = self._base_item(idx)
        name = seq.name

        onehot = F.one_hot(
            torch.load(os.path.join(_FOLD_SEEK_DIR, f"{name}.pt")).long(), num_classes=21
        )[0].unsqueeze(0).permute(0, 2, 1).reshape(21, -1).t()

        vector = torch.load(os.path.join(_ESM_IF_DIR, f"{name}.pt")).unsqueeze(0)
        vector = vector.permute(0, 2, 1).reshape(512, -1).t()

        return self._pack(seq, torch.cat([base_feat, vector, onehot], 1))


class PDB_esm(BasePDB):
    def __getitem__(self, idx):
        seq, feat = self._base_item(idx)
        return self._pack(seq, feat)


class PDB_saport(BasePDB):
    def __getitem__(self, idx):
        seq = self.data[idx]
        vector = torch.load(os.path.join(_SAPORT_DIR, f"{seq.name}.pt")).squeeze()[1:-1]
        return self._pack(seq, torch.cat([vector, seq.dssp], 1))


class PDB_structure(BasePDB):
    def __getitem__(self, idx):
        seq = self.data[idx]
        vector = torch.Tensor(
            np.load(os.path.join(f"{_STRUCT_BASE}/{self.sub_dir}", f"{seq.name}.feat_esm.npz"))["f"]
        ).unsqueeze(0).permute(0, 2, 1).reshape(640, -1).t()
        return self._pack(seq, torch.cat([seq.dssp, vector], 1))


class PDB_esm_structure(BasePDB):
    def __getitem__(self, idx):
        seq, base_feat = self._base_item(idx)
        vector = torch.Tensor(
            np.load(os.path.join(f"{_STRUCT_BASE}/{self.sub_dir}", f"{seq.name}.feat_esm.npz"))["f"]
        ).unsqueeze(0).permute(0, 2, 1).reshape(640, -1).t()
        return self._pack(seq, torch.cat([base_feat, vector], 1))


class PDB_DPO(BasePDB):
    """
    Dataset for DPO training.
    Returns coord and rsa in addition to the standard fields so that
    train_dpo.py can build preference pairs on-the-fly.
    """
    def __getitem__(self, idx):
        seq, feat = self._base_item(idx)
        return {**self._pack(seq, feat), 'coord': seq.coord, 'rsa': seq.rsa}


class PDB_DPO_gangxu(BasePDB):
    """
    Dataset for DPO training with GangXu FoldSeek tokens.
    Combines ESM-2 + DSSP + FoldSeek tokens from gang_xu directory.
    """
    def __getitem__(self, idx):
        seq, base_feat = self._base_item(idx)
        vector = torch.load(os.path.join(_FOLD_SEEK_GANGXU, f"{seq.name}.pt")).unsqueeze(0)
        onehot = F.one_hot(vector.long(), num_classes=21).permute(0, 2, 1).reshape(21, -1).t()
        feat = torch.cat([base_feat, onehot], 1)
        return {**self._pack(seq, feat), 'coord': seq.coord, 'rsa': seq.rsa}


def collate_fn_dpo(batch):
    """
    Collate for DPO training.
    Returns:
        feats       – list of feat tensors (variable length)
        edges       – list of edge tensors
        labels      – (N_total,) concatenated ground-truth labels
        coords_list – list of (N_i, 3) Cα coordinate tensors
        rsa_list    – list of (N_i,) RSA bool tensors
    """
    return (
        [item['feat']  for item in batch],
        [item['edge']  for item in batch],
        torch.cat([item['label'] for item in batch], 0),
        [item['coord'] for item in batch],
        [item['rsa']   for item in batch],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/BCE_633_fold_seek_esm_t_33_new')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    root = args.root
    device = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'

    os.system(f'cd {root} && mkdir feat dssp graph')
    model, _ = esm.pretrained.esm2_t33_650M_UR50D()

    state_dict = torch.load(
        "/work/home/maben/project/epitope_prediction/GraphBepi/esm2_1_249999_weight_cleaned.pt",
        map_location="cpu"
    )
    model.load_state_dict(state_dict, strict=False)
    print("Loaded local weights successfully!")

    model = model.to(device).eval()
    initial('total.csv', root, model, device)

    with open(f'{root}/total.pkl', 'rb') as f:
        dataset = pk.load(f)
    with open(f'{root}/date.pkl', 'rb') as f:
        dates = pk.load(f)

    filt_data = [i for i in dataset if len(i) < 1024 and i.label.sum() > 0]
    month = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
             'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    trainset, testset, dates_ = [], [], []
    test_cutoff = 20210401
    for i in tqdm(filt_data):
        d, m, y = dates[i.name[:-2]]
        d, m, y = int(d), month[m], int(y)
        y += 2000 if y < 23 else 1900
        date = y*10000 + m*100 + d
        if date < test_cutoff:
            dates_.append(date)
            trainset.append(i)
        else:
            testset.append(i)

    with open(f'{root}/train.pkl', 'wb') as f:
        pk.dump(trainset, f)
    with open(f'{root}/test.pkl', 'wb') as f:
        pk.dump(testset, f)
    np.save(f'{root}/cross-validation.npy', np.array(dates_).argsort())
