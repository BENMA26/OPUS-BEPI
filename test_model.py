import os
import torch
import warnings
import pytorch_lightning as pl
from tool import METRICS
from model import GraphBepi
from dataset import PDB, collate_fn
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from train_utils import seed_everything, build_arg_parser
warnings.simplefilter('ignore')

args = build_arg_parser().parse_args()
seed_everything(args.seed)
device = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'
root = f'./data/{args.dataset}'
log_name = f'{args.dataset}_{args.tag}'

trainset = PDB(mode='train', fold=args.fold, root=root)
valset   = PDB(mode='val',   fold=args.fold, root=root)
testset  = PDB(mode='test',  fold=args.fold, root=root)

train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader   = DataLoader(valset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(testset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
if args.fold == -1:
    val_loader = test_loader

model = GraphBepi(
    feat_dim=2560, hidden_dim=args.hidden, exfeat_dim=13, edge_dim=51,
    augment_eps=0.05, dropout=0.2, lr=args.lr,
    metrics=METRICS(device), result_path=f'./model/{log_name}',
)

mc = ModelCheckpoint(f'./model/{log_name}/', f'model_{args.fold}',
                     'val_AUPRC', mode='max', save_weights_only=True)
es = EarlyStopping('val_AUPRC', patience=40, mode='max')
logger = TensorBoardLogger(args.logger, name=f'{log_name}_{args.fold}')

accelerator = 'gpu' if args.gpu != -1 else 'cpu'
trainer = pl.Trainer(accelerator=accelerator, max_epochs=args.epochs,
                     callbacks=[mc, es], logger=logger, check_val_every_n_epoch=1)

model.load_state_dict(
    torch.load(f'./model/{log_name}/model_{args.fold}.ckpt')['state_dict']
)
trainer = pl.Trainer(accelerator=accelerator, logger=None)
result = trainer.test(model, test_loader)
os.rename(f'./model/{log_name}/result.pkl', f'./model/{log_name}/result_{args.fold}.pkl')
