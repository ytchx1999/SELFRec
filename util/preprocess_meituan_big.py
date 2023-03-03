import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict


def preprocess(train_data_name, val_data_name, test_data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    u2i_map_list = defaultdict(list)
    i2u_map_list = defaultdict(list)

    u_index_list = []
    i_index_list = []

    u_map = {}
    i_map = {}

    u_ind = 0
    i_ind = 0

    user_count = defaultdict(int)
    item_count = defaultdict(int)

    with open(train_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            click = int(e[2])
            if click == 0:
                continue
            user_count[u] += 1
            item_count[i] += 1
        
    with open(val_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            click = int(e[2])
            if click == 0:
                continue
            user_count[u] += 1
            item_count[i] += 1
    
    with open(test_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            click = int(e[2])
            if click == 0:
                continue
            user_count[u] += 1
            item_count[i] += 1

    num_interaction_train = 0
    num_interaction_val = 0
    num_interaction_test = 0

    with open(train_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            click = int(e[2])
            if click == 0:
                continue
            if user_count[u] < 10 or item_count[i] < 10:
                continue

            num_interaction_train += 1

            if u not in u_map:
                u_map[u] = u_ind
                u_ind += 1
            if i not in i_map:
                i_map[i] = i_ind
                i_ind += 1
            
            u = u_map[u]
            i = i_map[i]

            ts = float(e[5])
            label = float(1.)  # int(e[3])

            # feat = np.array([float(x) for x in e[4:]])

            # u2i_map_list[u].append(i)
            # i2u_map_list[i].append(u)

            # u_index_list.append(len(u2i_map_list[u]) - 1)
            # i_index_list.append(len(i2u_map_list[i]) - 1)

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
    
    with open(val_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            click = int(e[2])
            if click == 0:
                continue
            if user_count[u] < 10 or item_count[i] < 10:
                continue

            num_interaction_val += 1

            if u not in u_map:
                u_map[u] = u_ind
                u_ind += 1
            if i not in i_map:
                i_map[i] = i_ind
                i_ind += 1
            
            u = u_map[u]
            i = i_map[i]

            ts = float(e[5])
            label = float(1.)  # int(e[3])

            # feat = np.array([float(x) for x in e[4:]])

            # u2i_map_list[u].append(i)
            # i2u_map_list[i].append(u)

            # u_index_list.append(len(u2i_map_list[u]) - 1)
            # i_index_list.append(len(i2u_map_list[i]) - 1)

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

    with open(test_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            click = int(e[2])
            if click == 0:
                continue
            if user_count[u] < 10 or item_count[i] < 10:
                continue

            num_interaction_test += 1

            if u not in u_map:
                u_map[u] = u_ind
                u_ind += 1
            if i not in i_map:
                i_map[i] = i_ind
                i_ind += 1
            
            u = u_map[u]
            i = i_map[i]

            ts = float(e[5])
            label = float(1.)  # int(e[3])

            # feat = np.array([float(x) for x in e[4:]])

            # u2i_map_list[u].append(i)
            # i2u_map_list[i].append(u)

            # u_index_list.append(len(u2i_map_list[u]) - 1)
            # i_index_list.append(len(i2u_map_list[i]) - 1)

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
    
    num_interaction = num_interaction_train + num_interaction_val + num_interaction_test
    print('edge: ', num_interaction)
    print(float(num_interaction_train) / float(num_interaction))
    print(float(num_interaction_train + num_interaction_val) / float(num_interaction))

    df = pd.DataFrame({'u': u_list, 
                        'i':i_list, 
                        'ts':ts_list, 
                        'label':label_list, 
                        'idx':idx_list})
    
    df.sort_values("ts", inplace=True)

    users = df.u.values
    items = df.i.values

    for (u, i) in zip(users, items):
        u2i_map_list[u].append(i)
        i2u_map_list[i].append(u)

        u_index_list.append(len(u2i_map_list[u]) - 1)
        i_index_list.append(len(i2u_map_list[i]) - 1)
    
    df['u_idx'] = u_index_list
    df['i_idx'] = i_index_list
            # feat_l.append(feat)
    return df #, np.array(feat_l)


def reindex(df, bipartite=False):
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        print(new_df.u.max())
        print(new_df.i.max())

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


def run(data_name, bipartite=False):
    Path(f"./dataset/{data_name}").mkdir(parents=True, exist_ok=True)
    train_path = './dataset/{}/big_train.csv'.format(data_name, data_name)
    val_path = './dataset/{}/big_val.csv'.format(data_name, data_name)
    test_path = './dataset/{}/big_test.csv'.format(data_name, data_name)
    OUT_DF = './dataset/{}/ml_{}.csv'.format(data_name, data_name)
    OUT_FEAT = './dataset/{}/ml_{}.npy'.format(data_name, data_name)
    OUT_NODE_FEAT = './dataset/{}/ml_{}_node.npy'.format(data_name, data_name)
    train_path_txt = './dataset/{}/train.txt'.format(data_name)
    test_path_txt = './dataset/{}/test.txt'.format(data_name)

    df = preprocess(train_path, val_path, test_path)  # , feat
    # df.sort_values("ts", inplace=True)
    new_df = reindex(df, bipartite)

    # empty = np.zeros(feat.shape[1])[np.newaxis, :]
    # feat = np.vstack([empty, feat])

    # max_idx = max(new_df.u.max(), new_df.i.max())
    # rand_feat = np.zeros((max_idx + 1, 172))

    new_df.to_csv(OUT_DF)
    # np.save(OUT_FEAT, feat)
    # np.save(OUT_NODE_FEAT, rand_feat)
    ### Load data and train val test split
    graph_df = pd.read_csv(OUT_DF)

    val_time, test_time = list(np.quantile(graph_df.ts, [0.7137918444125029, 0.8570053042592756]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    # u_map = {}
    # i_map = {}
    # u_ind = 0
    # i_ind = 0
    # for (u, i) in zip(sources, destinations):
    #     if u not in u_map:
    #         u_map[u] = u_ind
    #         u_ind += 1
    #     if i not in i_map:
    #         i_map[i] = i_ind
    #         i_ind += 1

    user_item_src = []
    user_item_dst = []

    valid_train_flag = (timestamps <= val_time)
    
    # train
    user_item_src = sources[valid_train_flag]
    user_item_dst = destinations[valid_train_flag]
    train_e_idx_l = edge_idxs[valid_train_flag]

    valid_train_userset = set(np.unique(user_item_src))
    valid_train_itemset = set(np.unique(user_item_dst))
    
    # select validation and test dataset
    valid_val_flag = (timestamps <= test_time) * (timestamps > val_time)
    valid_test_flag = timestamps > test_time

    # validation and test with all edges
    val_src_l = sources[valid_val_flag]
    val_dst_l = destinations[valid_val_flag]
    
    valid_is_old_node_edge = np.array([(a in valid_train_userset and b in valid_train_itemset) for a, b in zip(val_src_l, val_dst_l)])
    val_src_ln = val_src_l[valid_is_old_node_edge]
    val_dst_ln = val_dst_l[valid_is_old_node_edge]

    # valid data -- Data
    print('#interactions in valid: ', len(val_src_ln))

    test_src_l = sources[valid_test_flag]
    test_dst_l = destinations[valid_test_flag]
    
    test_is_old_node_edge = np.array([(a in valid_train_userset and b in valid_train_itemset) for a, b in zip(test_src_l, test_dst_l)])
    test_src_ln = test_src_l[test_is_old_node_edge]
    test_dst_ln = test_dst_l[test_is_old_node_edge]

    # test data -- Data
    print('#interaction in test: ', len(test_src_ln))

    # train_items = defaultdict(list) # [[] for _ in range(user_item_src.max() + 1)]
    # for src, dst in zip(user_item_src, user_item_dst):
    #     train_items[src].append(dst)
    
    # test_set = defaultdict(list) # [[] for _ in range(test_src_ln.max() + 1)]
    # for src, dst in zip(test_src_ln, test_dst_ln):
    #     test_set[src].append(dst)

    f = open(train_path_txt, 'w')
    for src, dst in zip(user_item_src, user_item_dst):
        f.write(str(src) + ' ' + str(dst) + ' ' + str(1) + '\n')
    f.close()
    
    f = open(test_path_txt, 'w')
    for src, dst in zip(val_src_ln, val_dst_ln):
        f.write(str(src) + ' ' + str(dst) + ' ' + str(1) + '\n')
    f.close()


parser = argparse.ArgumentParser('Interface for LSTDR data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='meituan_big')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)
