import warnings
warnings.simplefilter('ignore')
from tool import METRICS
from model import GraphBepi, GraphBepi_att
from dataset import (
    PDB, PDB_esm, PDB_token_esm, PDB_saport,
    PDB_foldseek, PDB_foldseek_local_golbal, PDB_foldseek_attn,
    PDB_esm_if, PDB_esm_if_foldseek_tokens,
    PDB_structure, PDB_esm_structure,
    collate_fn, collate_fn_fold_tokens,
)
from train_utils import seed_everything, build_arg_parser, run_training

# Registry: mode -> (dataset_class, feat_dim, model_class, collate_fn, use_early_stop, needs_sub_dir)
CONFIGS = {
    'esm2_3b':        (PDB,                          2560,        GraphBepi,     collate_fn,             False, False),
    'esm2_650m':      (PDB,                          1280,        GraphBepi,     collate_fn,             True,  False),
    'esm2_3b_es':     (PDB,                          2560,        GraphBepi,     collate_fn,             True,  False),
    'esm_t':          (PDB_token_esm,                2560+640,    GraphBepi,     collate_fn,             False, False),
    'saport':         (PDB_saport,                   446,         GraphBepi,     collate_fn,             False, False),
    'esm2_gangxu':    (PDB_esm,                      2560,        GraphBepi,     collate_fn,             False, False),
    'structure':      (PDB_structure,                640,         GraphBepi,     collate_fn,             False, True),
    'esm2_structure': (PDB_esm_structure,            2560+640,    GraphBepi,     collate_fn,             False, True),
    'saport_gangxu':  (PDB_saport,                   446,         GraphBepi,     collate_fn,             False, False),
    'foldseek_multi': (PDB_foldseek,                 2686,        GraphBepi,     collate_fn,             True,  False),
    'foldseek_single':(PDB_foldseek_local_golbal,    2581+21,     GraphBepi,     collate_fn,             False, False),
    'foldseek_attn':  (PDB_foldseek_attn,            2581,        GraphBepi_att, collate_fn_fold_tokens, True,  False),
    'esm_if':         (PDB_esm_if,                   2560+512,    GraphBepi,     collate_fn,             False, False),
    'esm_if_foldseek':(PDB_esm_if_foldseek_tokens,   2560+512+21, GraphBepi,     collate_fn,             False, False),
}

parser = build_arg_parser()
parser.add_argument('--mode', type=str, required=True, choices=CONFIGS,
                    help='feature configuration to use.')
parser.add_argument('--sub_dir', type=str, default='BCE_633',
                    help='structure feature sub-directory (used by structure/esm2_structure modes).')
args = parser.parse_args()
seed_everything(args.seed)

dataset_cls, feat_dim, model_cls, cfn, use_es, needs_sub_dir = CONFIGS[args.mode]
device   = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'
root     = f'./data/{args.dataset}'
log_name = f'{args.dataset}_{args.tag}'

ds_kwargs = {'sub_dir': args.sub_dir} if needs_sub_dir else {}
trainset = dataset_cls(mode='train', fold=args.fold, root=root, **ds_kwargs)
valset   = dataset_cls(mode='val',   fold=args.fold, root=root, **ds_kwargs)
testset  = dataset_cls(mode='test',  fold=args.fold, root=root, **ds_kwargs)

model = model_cls(
    feat_dim=feat_dim, hidden_dim=args.hidden, exfeat_dim=13, edge_dim=51,
    augment_eps=0.05, dropout=0.2, lr=args.lr,
    metrics=METRICS(device), result_path=f'./model/{log_name}',
)
run_training(model, trainset, valset, testset, args, cfn, use_early_stop=use_es)
