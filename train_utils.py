import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
warnings.simplefilter('ignore')


def seed_everything(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate.')
    parser.add_argument('--gpu', type=int, default=0, help='gpu.')
    parser.add_argument('--fold', type=int, default=-1, help='dataset fold. -1 = whole trainset.')
    parser.add_argument('--seed', type=int, default=2022, help='random seed.')
    parser.add_argument('--batch', type=int, default=4, help='batch size.')
    parser.add_argument('--hidden', type=int, default=256, help='hidden dim.')
    parser.add_argument('--epochs', type=int, default=300, help='max number of epochs.')
    parser.add_argument('--dataset', type=str, default='BCE_633', help='dataset name.')
    parser.add_argument('--logger', type=str, default='./log', help='logger path.')
    parser.add_argument('--tag', type=str, default='GraphBepi', help='logger name.')
    # ── Loss function ──────────────────────────────────────────────────────────
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'afl', 'bce_dice', 'smooth_bce', 'pu'],
                        help='Loss function. bce=baseline, afl=AsymmetricFocal, '
                             'bce_dice=BCE+Dice, smooth_bce=LabelSmoothing, pu=nnPU.')
    # AFL params
    parser.add_argument('--afl_gamma_pos', type=float, default=0.0,
                        help='[AFL] focusing exponent for positives (default 0 = no focusing).')
    parser.add_argument('--afl_gamma_neg', type=float, default=2.0,
                        help='[AFL] focusing exponent for negatives.')
    parser.add_argument('--afl_clip', type=float, default=0.05,
                        help='[AFL] probability shift margin for negatives.')
    # BCE+Dice params
    parser.add_argument('--dice_alpha', type=float, default=0.5,
                        help='[BCE+Dice] weight of BCE term (0=pure Dice, 1=pure BCE).')
    # Label-smoothing params
    parser.add_argument('--smooth_eps', type=float, default=0.05,
                        help='[smooth_bce] positive-side smoothing epsilon.')
    # PU params
    parser.add_argument('--pu_prior', type=float, default=0.15,
                        help='[pu] prior P(y=1); typical epitope fraction ~0.15.')
    return parser


def build_loss_fn(args):
    """Construct the loss function from parsed args."""
    from losses import AsymmetricFocalLoss, BCEDiceLoss, LabelSmoothingBCELoss, PULoss
    if args.loss == 'bce':
        return nn.BCELoss()
    elif args.loss == 'afl':
        return AsymmetricFocalLoss(
            gamma_pos=args.afl_gamma_pos,
            gamma_neg=args.afl_gamma_neg,
            clip=args.afl_clip,
        )
    elif args.loss == 'bce_dice':
        return BCEDiceLoss(alpha=args.dice_alpha)
    elif args.loss == 'smooth_bce':
        return LabelSmoothingBCELoss(eps=args.smooth_eps)
    elif args.loss == 'pu':
        return PULoss(prior=args.pu_prior)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


def run_training(model, trainset, valset, testset, args, collate_fn, use_early_stop=False):
    log_name = f'{args.dataset}_{args.tag}'
    accelerator = 'gpu' if args.gpu != -1 else 'cpu'

    train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(testset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    if args.fold == -1:
        val_loader = test_loader

    mc = ModelCheckpoint(f'./model/{log_name}/', f'model_{args.fold}',
                         'val_AUPRC', mode='max', save_weights_only=True)
    logger = TensorBoardLogger(args.logger, name=f'{log_name}_{args.fold}')
    callbacks = [mc]
    if use_early_stop:
        callbacks.append(EarlyStopping('val_AUPRC', patience=40, mode='max'))

    trainer = pl.Trainer(
        accelerator=accelerator, max_epochs=args.epochs,
        callbacks=callbacks, logger=logger, check_val_every_n_epoch=1,
    )

    ckpt_path = f'./model/{log_name}/model_{args.fold}.ckpt'
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    trainer.fit(model, train_loader, val_loader)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    trainer = pl.Trainer(accelerator=accelerator, logger=logger)
    trainer.test(model, test_loader)
    os.rename(f'./model/{log_name}/result.pkl', f'./model/{log_name}/result_{args.fold}.pkl')
