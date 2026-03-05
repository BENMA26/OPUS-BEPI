import os
import torch
import numpy as np
import pandas as pd
import pickle as pk
from tqdm import tqdm
from preprocess import *
from graph_construction import calcPROgraph

amino2id = {
    '<null_0>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10,
    'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16,
    'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22,
    'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28,
    '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32, '<cath>': 33, '<af2>': 34
}


class chain:
    def __init__(self):
        self.sequence = []
        self.amino = []
        self.coord = []
        self.site = {}
        self.date = ''
        self.length = 0
        self.adj = None
        self.edge = None
        self.feat = None
        self.dssp = None
        self.name = ''
        self.chain_name = ''
        self.protein_name = ''

    def add(self, amino, pos, coord):
        self.sequence.append(DICT[amino])
        self.amino.append(amino2id[DICT[amino]])
        self.coord.append(coord)
        self.site[pos] = self.length
        self.length += 1

    def process(self):
        self.amino = torch.LongTensor(self.amino)
        self.coord = torch.FloatTensor(self.coord)
        self.label = torch.zeros_like(self.amino)
        self.sequence = ''.join(self.sequence)

    def extract(self, model, device, path):
        if len(self) > 1024 or model is None:
            return
        with torch.no_grad():
            feat = model(self.amino.to(device).unsqueeze(0), [33])['representations'][33].squeeze(0).cpu()
        torch.save(feat, f'{path}/feat/{self.name}_esm2.ts')

    def load_dssp(self, path):
        dssp = torch.Tensor(np.load(f'{path}/dssp/{self.name}.npy'))
        pos = np.load(f'{path}/dssp/{self.name}_pos.npy')
        self.dssp = torch.Tensor([
            -2.4492936e-16, -2.4492936e-16,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]).repeat(self.length, 1)
        self.rsa = torch.zeros(self.length)
        for i in range(len(dssp)):
            self.dssp[self.site[pos[i]]] = dssp[i]
            if dssp[i][4] > 0.15:
                self.rsa[i] = 1
        self.rsa = self.rsa.bool()

    def load_feat(self, path):
        self.feat = torch.load(f'{path}/feat/{self.name}_esm2.ts')

    def load_adj(self, path, self_cycle=False):
        graph = torch.load(f'{path}/graph/{self.name}.graph')
        self.adj = graph['adj'].to_dense()
        self.edge = graph['edge'].to_dense()
        if not self_cycle:
            self.adj[range(len(self)), range(len(self))] = 0
            self.edge[range(len(self)), range(len(self))] = 0

    def get_adj(self, path, dseq=3, dr=10, dlong=5, k=10):
        graph = calcPROgraph(self.sequence, self.coord, dseq, dr, dlong, k)
        torch.save(graph, f'{path}/graph/{self.name}.graph')

    def update(self, pos, amino):
        if amino not in DICT.keys():
            return
        amino_id = amino2id[DICT[amino]]
        idx = self.site.get(pos, None)
        if idx is None:
            for i in self.site.keys():
                if i[:len(pos)] == pos:
                    idx = self.site.get(i)
                    if amino_id == self.amino[idx]:
                        self.label[idx] = 1
                        return
        elif amino_id != self.amino[idx]:
            for i in self.site.keys():
                if i[:len(pos)] == pos:
                    idx = self.site.get(i)
                    if amino_id == self.amino[idx]:
                        self.label[idx] = 1
                        return
        else:
            self.label[idx] = 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.amino[idx], self.coord[idx], self.label[idx]


def collate_fn(batch):
    return (
        [item['feat'] for item in batch],
        [item['edge'] for item in batch],
        torch.cat([item['label'] for item in batch], 0),
    )


def collate_fn_fold_tokens(batch):
    return (
        [item['fold_token'] for item in batch],
        [item['feat'] for item in batch],
        [item['edge'] for item in batch],
        torch.cat([item['label'] for item in batch], 0),
    )


def extract_chain(root, pid, chain_id, force=False):
    if not force and os.path.exists(f'{root}/purePDB/{pid}_{chain_id}.pdb'):
        return True
    if not os.path.exists(f'{root}/PDB/{pid}.pdb'):
        import requests as rq
        pdb = None
        for _ in range(5):
            try:
                with rq.get(f'https://files.wwpdb.org/pub/pdb/data/structures/all/pdb/{pid}.pdb') as f:
                    if f.status_code == 200:
                        pdb = f.content
                        break
            except Exception:
                continue
        if pdb is None:
            print(f'PDB file {pid} failed to download')
            return False
        with open(f'{root}/PDB/{pid}.pdb', 'wb') as f:
            f.write(pdb)
    lines = []
    with open(f'{root}/PDB/{pid}.pdb', 'r') as f:
        for line in f:
            if line[:6] == 'HEADER':
                lines.append(line)
            if line[:6].strip() == 'TER' and line[21] == chain_id:
                lines.append(line)
                break
            feats = judge(line, None)
            if feats is not None and feats[1] == chain_id:
                lines.append(line)
    with open(f'{root}/purePDB/{pid}_{chain_id}.pdb', 'w') as f:
        f.writelines(lines)
    return True


def process_chain(data, root, pid, model, device):
    get_dssp(pid, root)
    same = {}
    with open(f'{root}/purePDB/{pid}.pdb', 'r') as f:
        for line in f:
            if line[:6] == 'HEADER':
                data.date = line[50:59].strip()
                continue
            feats = judge(line, 'CA')
            if feats is None:
                continue
            amino, _, site, x, y, z = feats
            if len(amino) > 3:
                if same.get(site) is None:
                    same[site] = amino[0]
                if same[site] != amino[0]:
                    continue
                amino = amino[-3:]
            data.add(amino, site, [x, y, z])
    data.process()
    data.get_adj(root)
    data.extract(model, device, root)
    return data


def initial(file, root, model=None, device='cpu', from_native_pdb=True):
    df = pd.read_csv(f'{root}/{file}', header=0, index_col=0)
    labels = df['Epitopes (resi_resn)']
    samples = []
    with tqdm(df.index) as tbar:
        for i in tbar:
            tbar.set_postfix(protein=i)
            if from_native_pdb:
                if not extract_chain(root, i[:4], i[-1]):
                    continue
            data = chain()
            p, c = i.split('_')
            data.protein_name, data.chain_name, data.name = p, c, f"{p}_{c}"
            process_chain(data, root, i, model, device)
            for j in labels.loc[i].split(', '):
                site, amino = j.split('_')
                data.update(site, amino)
            samples.append(data)
    with open(f'{root}/total.pkl', 'wb') as f:
        pk.dump(samples, f)
