import os
import esm
import torch
import warnings
import argparse
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
warnings.simplefilter('ignore')
class PDB(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        seq=self.data[idx]
        feat=torch.cat([seq.feat,seq.dssp],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }

class PDB_foldseek(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633_fold_seek',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        seq=self.data[idx]
        name=self.data[idx].name
        vector_name=f"{name}.pt"
        vector_path=os.path.join("/work/home/maben/project/epitope_prediction/GraphBepi/data/fold_seek_vectors",vector_name)
        vector = torch.load(vector_path)
        onehot = torch.nn.functional.one_hot(vector.to(torch.long), num_classes=21)
        onehot = onehot.permute(0, 2, 1).reshape(21*6, -1).transpose(0,1)
        feat=torch.cat([seq.feat,seq.dssp,onehot],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }

class PDB_foldseek_local_golbal(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633_fold_seek',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        seq=self.data[idx]
        name=self.data[idx].name
        vector_name=f"{name}.pt"
        vector_path=os.path.join("/work/home/maben/project/epitope_prediction/GraphBepi/data/fold_seek_vectors_gang_xu",vector_name)
        vector = torch.load(vector_path).unsqueeze(0)
        onehot = torch.nn.functional.one_hot(vector.to(torch.long), num_classes=21)
        onehot_global = onehot.permute(0, 2, 1).reshape(21, -1).transpose(0,1)

        vector_path=os.path.join("/work/home/maben/project/epitope_prediction/GraphBepi/data/fold_seek_vectors",vector_name)
        vector = torch.load(vector_path)[0].unsqueeze(0)
        onehot = torch.nn.functional.one_hot(vector.to(torch.long), num_classes=21)
        onehot_local = onehot.permute(0, 2, 1).reshape(21, -1).transpose(0,1)

        feat=torch.cat([seq.feat,seq.dssp,onehot_global,onehot_local],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }

class PDB_foldseek_attn(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633_fold_seek',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        seq=self.data[idx]
        name=self.data[idx].name
        vector_name=f"{name}.pt"
        vector_path=os.path.join("/work/home/maben/project/epitope_prediction/GraphBepi/data/fold_seek_vectors",vector_name)
        vector = torch.load(vector_path)
        onehot = torch.nn.functional.one_hot(vector.to(torch.long), num_classes=21)
        onehot = onehot.permute(0, 2, 1).reshape(-1,21,6)
        feat=torch.cat([seq.feat,seq.dssp],1)
        return {
            'feat':feat,
            'fold_token':onehot,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }
        
class PDB_foldseek_tokens(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633_fold_seek',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        seq=self.data[idx]
        name=self.data[idx].name
        vector_name=f"{name}.pt"
        #vector_path=os.path.join("/work/home/maben/project/epitope_prediction/GraphBepi/data/fold_seek_vectors",vector_name)
        vector_path=os.path.join("/work/home/maben/project/epitope_prediction/GraphBepi/data/fold_seek_vectors_gang_xu",vector_name)
        #vector = torch.load(vector_path)[0].unsqueeze(0)
        vector = torch.load(vector_path).unsqueeze(0)#[0].unsqueeze(0)
        onehot = torch.nn.functional.one_hot(vector.to(torch.long), num_classes=21)
        onehot = onehot.permute(0, 2, 1).reshape(21, -1).transpose(0,1)
        #print(onehot.shape)
        feat=torch.cat([seq.feat,seq.dssp,onehot],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }

class PDB_token_esm(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633_fold_seek',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        seq=self.data[idx]
        name=self.data[idx].name
        vector_name=f"{name}.pt"
        vector = np.load(os.path.join("/work/home/xugang/projects/struct_diff/maben/feat_esm", name + ".feat_esm.npz"))["f"]
        vector = torch.Tensor(vector)
        #onehot = F.layer_norm(vector, normalized_shape=[640], eps=1e-5)  # shape: (L, 640)
        onehot = vector.unsqueeze(0)
        onehot = onehot.permute(0, 2, 1).reshape(640, -1).transpose(0,1)
        feat=torch.cat([seq.feat,seq.dssp,onehot],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }

class PDB_esm_if(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633_fold_seek',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        seq=self.data[idx]
        name=self.data[idx].name
        vector_name=f"{name}.pt"
        vector_path=os.path.join("/work/home/maben/project/epitope_prediction/GraphBepi/data/BCE_633_fold_seek/esm-if",vector_name)
        vector = torch.load(vector_path).unsqueeze(0)
        vector = vector.permute(0, 2, 1).reshape(512, -1).transpose(0,1)
        feat=torch.cat([seq.feat,seq.dssp,vector],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }

class PDB_esm_if_foldseek_tokens(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633_fold_seek',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        seq=self.data[idx]
        name=self.data[idx].name
        vector_name=f"{name}.pt"

        vector_path=os.path.join("/work/home/maben/project/epitope_prediction/GraphBepi/data/fold_seek_vectors",vector_name)
        vector = torch.load(vector_path)
        onehot = torch.nn.functional.one_hot(vector.to(torch.long), num_classes=21)[0].unsqueeze(0)
        onehot = onehot.permute(0, 2, 1).reshape(21, -1).transpose(0,1)

        vector_path=os.path.join("/work/home/maben/project/epitope_prediction/GraphBepi/data/BCE_633_fold_seek/esm-if",vector_name)
        vector = torch.load(vector_path).unsqueeze(0)
        vector = vector.permute(0, 2, 1).reshape(512, -1).transpose(0,1)
        feat=torch.cat([seq.feat,seq.dssp,vector,onehot],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }

'''
experimental codes for gangxu
'''
class PDB_esm(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633_fold_seek',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        seq=self.data[idx]
        feat=torch.cat([seq.feat,seq.dssp],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }

class PDB_saport(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633_fold_seek',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        seq=self.data[idx]
        name=self.data[idx].name
        vector_name=f"{name}.pt"
        vector_path=os.path.join("/work/home/maben/project/epitope_prediction/SaProt/get_inputs/saport_embeddings",vector_name)
        vector = torch.load(vector_path).squeeze()[1:-1]
        feat=torch.cat([vector,seq.dssp],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }

class PDB_structure(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633_fold_seek',self_cycle=False,sub_dir=None
    ):
        self.sub_dir = sub_dir
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        seq=self.data[idx]
        name=self.data[idx].name
        vector_name=f"{name}.pt"
        #vector = np.load(os.path.join("/work/home/xugang/projects/struct_diff/maben", name + ".feat_esm.npz"))["f"]
        vector = np.load(os.path.join(f"/work/home/xugang/projects/struct_diff/maben/{self.sub_dir}", name + ".feat_esm.npz"))["f"]
        vector = torch.Tensor(vector)
        onehot = vector.unsqueeze(0)
        onehot = onehot.permute(0, 2, 1).reshape(640, -1).transpose(0,1)
        feat=torch.cat([seq.dssp,onehot],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }

class PDB_esm_structure(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633_fold_seek',self_cycle=False,sub_dir=None
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])
        
        self.subdir = sub_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        seq=self.data[idx]
        name=self.data[idx].name
        vector_name=f"{name}.pt"
        #vector = np.load(os.path.join("/work/home/xugang/projects/struct_diff/maben/feat_esm", name + ".feat_esm.npz"))["f"]
        vector = np.load(os.path.join(f"/work/home/xugang/projects/struct_diff/maben/{self.subdir}", name + ".feat_esm.npz"))["f"]
        vector = torch.Tensor(vector)
        onehot = vector.unsqueeze(0)
        onehot = onehot.permute(0, 2, 1).reshape(640, -1).transpose(0,1)
        feat=torch.cat([seq.feat,seq.dssp,onehot],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/BCE_633_fold_seek_esm_t_33_new', help='dataset path')
    parser.add_argument('--gpu', type=int, default=0, help='gpu.')
    args = parser.parse_args()
    root = args.root
    device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
    
    #os.system(f'cd {root} && mkdir PDB purePDB feat dssp graph')
    os.system(f'cd {root} && mkdir feat dssp graph')
    # model=None
    model,_=esm.pretrained.esm2_t33_650M_UR50D()

    state_dict = torch.load("/work/home/maben/project/epitope_prediction/GraphBepi/esm2_1_249999_weight_cleaned.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    print("Loaded local weights successfully!")

    model=model.to(device)
    model.eval()
    train='total.csv'
    initial(train,root,model,device)
    with open(f'{root}/total.pkl','rb') as f:
        dataset=pk.load(f)
    dates={i.name:i.date for i in dataset}
    with open(f'{root}/date.pkl','rb') as f:
        dates=pk.load(f)
    filt_data=[]
    for i in dataset:
        if len(i)<1024 and i.label.sum()>0:
            filt_data.append(i)
    month={'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    trainset,valset,testset=[],[],[]
    D,M,Y=[],[],[]
    test=20210401
    dates_=[]
    for i in tqdm(filt_data):
        d,m,y=dates[i.name[:-2]]
        d,m,y=int(d),month[m],int(y)
        if y<23:
            y+=2000
        else:
            y+=1900
        date=y*10000+m*100+d
        if date<test:
            dates_.append(date)
            trainset.append(i)
        else:
            testset.append(i)
    with open(f'{root}/train.pkl','wb') as f:
        pk.dump(trainset,f)
    with open(f'{root}/test.pkl','wb') as f:
        pk.dump(testset,f)
    idx=np.array(dates_).argsort()
    np.save(f'{root}/cross-validation.npy',idx)
