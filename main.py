from SELFRec import SELFRec
from util.conf import ModelConf
import random
import numpy as np
import torch
import argparse

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # Register your model here
    parser = argparse.ArgumentParser('SELFRec')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--seed', type=int, default=0, help='Seed for all')
    parser.add_argument('--model', type=str, default="SGL", choices=['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL'], help='Type of module')

    args = parser.parse_args()
    set_seed(args.seed)
    
    baseline = ['LightGCN','MF']
    graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL']
    sequential_models = []

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print('=' * 80)

    print('Baseline Models:')
    print('   '.join(baseline))
    print('-' * 80)
    print('Graph-Based Models:')
    print('   '.join(graph_models))

    print('=' * 80)
    model = args.model
    print(args.model)
    # model = input('Please enter the model you want to run:')
    import time

    if model in baseline or model in graph_models or model in sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)

    best_result_precision = []
    best_result_recall = []
    best_result_ndcg = []
    best_result_hit = []
    for run in range(args.n_runs):
        print('-'*50, flush=True)
        print(f'Run {run}:', flush=True)
        set_seed(args.seed + run)
        s = time.time()
        rec = SELFRec(conf)
        ans = rec.execute()
        e = time.time()
        print("Running time: %f s" % (e - s))

        best_result_precision.append(ans['precision'])
        best_result_recall.append(ans['recall'])
        best_result_ndcg.append(ans['ndcg'])
        best_result_hit.append(ans['hit'])

    best_result_precision_mean, best_result_precision_std = np.mean(np.array(best_result_precision), axis=0), np.std(np.array(best_result_precision), axis=0)
    best_result_recall_mean, best_result_recall_std = np.mean(np.array(best_result_recall), axis=0), np.std(np.array(best_result_recall), axis=0)
    best_result_ndcg_mean, best_result_ndcg_std = np.mean(np.array(best_result_ndcg), axis=0), np.std(np.array(best_result_ndcg), axis=0)
    best_result_hit_mean, best_result_hit_std = np.mean(np.array(best_result_hit), axis=0), np.std(np.array(best_result_hit), axis=0)

    print(f'Final test precision: {best_result_precision_mean} ± {best_result_precision_std}', flush=True)
    print(f'Final test recall: {best_result_recall_mean} ± {best_result_recall_std}', flush=True)
    print(f'Final test ndcg: {best_result_ndcg_mean} ± {best_result_ndcg_std}', flush=True)
    print(f'Final test hit_ratio: {best_result_hit_mean} ± {best_result_hit_std}', flush=True)
