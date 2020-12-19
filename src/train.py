import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]
sys.path.append(root_path + 'src')
os.chdir(root_path)
from early_stopper import *
from hin_loader import HIN
from evaluation import *
import util_funcs as uf
from config import HGSLConfig
from HGSL import HGSL
import warnings
import time
import torch
import argparse

warnings.filterwarnings('ignore')
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]


def train_hgsl(args, gpu_id=0, log_on=True):
    uf.seed_init(args.seed)
    uf.shell_init(gpu_id=gpu_id)
    cf = HGSLConfig(args.dataset)

    # ! Modify config
    cf.update(args.__dict__)
    cf.dev = torch.device("cuda:0" if gpu_id >= 0 else "cpu")

    # ! Load Graph
    g = HIN(cf.dataset).load_mp_embedding(cf)
    print(f'Dataset: {cf.dataset}, {g.t_info}')
    features, adj, mp_emb, train_x, train_y, val_x, val_y, test_x, test_y = g.to_torch(cf)

    # ! Train Init
    if not log_on: uf.block_logs()
    print(f'{cf}\nStart training..')
    cla_loss = torch.nn.NLLLoss()
    model = HGSL(cf, g)
    model.to(cf.dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file)

    dur = []
    w_list = []
    for epoch in range(cf.epochs):
        # ! Train
        t0 = time.time()
        model.train()
        logits, adj_new = model(features, adj, mp_emb)
        train_f1, train_mif1 = eval_logits(logits, train_x, train_y)
        w_list.append(uf.print_weights(model))

        l_pred = cla_loss(logits[train_x], train_y)
        l_reg = cf.alpha * torch.norm(adj, 1)
        loss = l_pred + l_reg
        optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            loss.backward()
        optimizer.step()

        # ! Valid
        model.eval()
        with torch.no_grad():
            logits = model.GCN(features, adj_new)
            val_f1, val_mif1 = eval_logits(logits, val_x, val_y)
        dur.append(time.time() - t0)
        uf.print_train_log(epoch, dur, loss, train_f1, val_f1)

        if cf.early_stop > 0:
            if stopper.step(val_f1, model, epoch):
                print(f'Early stopped, loading model from epoch-{stopper.best_epoch}')
                break

    if cf.early_stop > 0:
        model.load_state_dict(torch.load(cf.checkpoint_file))
    logits, _ = model(features, adj, mp_emb)
    cf.update(w_list[stopper.best_epoch])
    eval_and_save(cf, logits, test_x, test_y, val_x, val_y, stopper)
    if not log_on: uf.enable_logs()
    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dataset = 'yelp'
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train_hgsl(args, gpu_id=args.gpu_id)
