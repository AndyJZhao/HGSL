import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# * ============================= Init =============================
def shell_init(server='S5', gpu_id=0):
    '''

    Features:
    1. Specify server specific source and python command
    2. Fix Pycharm LD_LIBRARY_ISSUE
    3. Block warnings
    4. Block TF useless messages
    5. Set paths
    '''
    import warnings
    np.seterr(invalid='ignore')
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if server == 'Xy':
        python_command = '/home/chopin/zja/anaconda/bin/python'
    elif server == 'Colab':
        python_command = 'python'
    else:
        python_command = '~/anaconda3/bin/python'
        if gpu_id > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64/'  # Extremely useful for Pycharm users
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Block TF messages

    return python_command


def seed_init(seed):
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# * ============================= Torch =============================

def exists_zero_lines(h):
    zero_lines = torch.where(torch.sum(h, 1) == 0)[0]
    if len(zero_lines) > 0:
        # raise ValueError('{} zero lines in {}s!\nZero lines:{}'.format(len(zero_lines), 'emb', zero_lines))
        print(f'{len(zero_lines)} zero lines !\nZero lines:{zero_lines}')
        return True
    return False


def cos_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


# * ============================= Print Related =============================

def print_dict(d, end_string='\n\n'):
    for key in d.keys():
        if isinstance(d[key], dict):
            print('\n', end='')
            print_dict(d[key], end_string='')
        elif isinstance(d[key], int):
            print('{}: {:04d}'.format(key, d[key]), end=', ')
        elif isinstance(d[key], float):
            print('{}: {:.4f}'.format(key, d[key]), end=', ')
        else:
            print('{}: {}'.format(key, d[key]), end=', ')
    print(end_string, end='')


def block_logs():
    sys.stdout = open(os.devnull, 'w')
    logger = logging.getLogger()
    logger.disabled = True


def enable_logs():
    # Restore
    sys.stdout = sys.__stdout__
    logger = logging.getLogger()
    logger.disabled = False


def progress_bar(prefix, start_time, i, max_i, postfix):
    """
    Generates progress bar AFTER the ith epoch.
    Args:
        prefix: the prefix of printed string
        start_time: start time of the loop
        i: finished epoch index
        max_i: total iteration times
        postfix: the postfix of printed string

    Returns: prints the generated progress bar

    """
    cur_run_time = time.time() - start_time
    i += 1
    if i != 0:
        total_estimated_time = cur_run_time * max_i / i
    else:
        total_estimated_time = 0
    print(
        f'{prefix} :  {i}/{max_i} [{time2str(cur_run_time)}/{time2str(total_estimated_time)}, {time2str(total_estimated_time - cur_run_time)} left] - {postfix}-{get_cur_time()}')


def print_train_log(epoch, dur, loss, train_f1, val_f1):
    print(
        f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} | Loss {loss.item():.4f} | TrainF1 {train_f1:.4f} | ValF1 {val_f1:.4f}")


def mp_list_str(mp_list):
    return '_'.join(mp_list)


# * ============================= File Operations =============================

def write_nested_dict(d, f_path):
    def _write_dict(d, f):
        for key in d.keys():
            if isinstance(d[key], dict):
                f.write(str(d[key]) + '\n')

    with open(f_path, 'a+') as f:
        f.write('\n')
        _write_dict(d, f)


def save_pickle(var, f_name):
    pickle.dump(var, open(f_name, 'wb'))


def load_pickle(f_name):
    return pickle.load(open(f_name, 'rb'))


def clear_results(dataset, model):
    res_path = f'results/{dataset}/{model}/'
    os.system(f'rm -rf {res_path}')
    print(f'Results in {res_path} are cleared.')


# * ============================= Path Operations =============================

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def get_grand_parent_dir(f_name):
    if '.' in f_name.split('/')[-1]:  # File
        return get_grand_parent_dir(get_dir_of_file(f_name))
    else:  # Path
        return f'{Path(f_name).parent}/'


def get_abs_path(f_name, style='command_line'):
    # python 中的文件目录对空格的处理为空格，命令行对空格的处理为'\ '所以命令行相关需 replace(' ','\ ')
    if style == 'python':
        cur_path = os.path.abspath(os.path.dirname(__file__))
    elif style == 'command_line':
        cur_path = os.path.abspath(os.path.dirname(__file__)).replace(' ', '\ ')

    root_path = cur_path.split('src')[0]
    return os.path.join(root_path, f_name)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:

        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def mkdir_list(p_list, use_relative_path=True, log=True):
    """Create directories for the specified path lists.
        Parameters
        ----------
        p_list :Path lists

    """
    # ! Note that the paths MUST END WITH '/' !!!
    root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
    for p in p_list:
        p = os.path.join(root_path, p) if use_relative_path else p
        p = os.path.dirname(p)
        mkdir_p(p, log)


# * ============================= Time Related =============================

def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time():
    import datetime
    dt = datetime.datetime.now()
    return f'{dt.date()}_{dt.hour:02d}-{dt.minute:02d}-{dt.second:02d}'


# * ============================= Others =============================
def print_weights(model, interested_para='_agg'):
    w_dict = {}
    for name, W in model.named_parameters():
        if interested_para in name:
            data = F.softmax(W.data.squeeze()).cpu().numpy()
            # print(f'{name}:{data}')
            w_dict[name] = data
    return w_dict


def count_avg_neighbors(adj):
    return len(torch.where(adj > 0)[0]) / adj.shape[0]
