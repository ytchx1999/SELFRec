from tkinter.messagebox import NO
import numpy as np
import random
import pandas as pd


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels, u_index, i_index, u_start_idx=None, i_start_idx=None):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.u_index = u_index
        self.i_index = i_index
        self.u_start_idx = u_start_idx
        self.i_start_idx = i_start_idx
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data_node_classification(dataset_name, use_validation=False):
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
    node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values
    # u_index = graph_df.u_idx.values
    # i_index = graph_df.i_idx.values

    # random.seed(2020)

    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False, start_percent=0.0):
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    # u_ind_id_map = pd.read_csv('./data/{}/{}_u_map.json'.format(dataset_name, dataset_name))
    # i_ind_id_map = pd.read_csv('./data/{}/{}_i_map.json'.format(dataset_name, dataset_name))

    # edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
    # node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))
    no_edge_dataset = ["ml100k", "ml-1m", "lastfm", "music", "baby", "gowalla", "toys", "food", "garden", "video", "instruments", "auto", "magazine", "software", "meituan", "meituan_big", "pantry", "scientific", "beauty"]
    if dataset_name in no_edge_dataset:
        edge_features = None
        node_features = None
    else:
        edge_features = np.load('./data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_features = np.load('./data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    # if randomize_features:
    #     node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

    # val_time, test_time = list(np.quantile(graph_df.ts, [0.80, 0.90]))
    if dataset_name == 'meituan':
        val_time, test_time, test_time_2 = list(np.quantile(graph_df.ts, [0.5224476546015258, 0.7530271059892875, 0.92]))  # meituan
    elif dataset_name == 'meituan_big':
        val_time, test_time, test_time_2 = list(np.quantile(graph_df.ts, [0.7137918444125029, 0.8570053042592756, 0.92]))  # meituan
    else:
        val_time, test_time, test_time_2 = list(np.quantile(graph_df.ts, [0.80, 0.90, 0.92]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values
    u_index = graph_df.u_idx.values
    i_index = graph_df.i_idx.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels, u_index, i_index)

    random.seed(2022)

    # max_src_index = sources.max()
    max_src_index = sources.max()
    max_idx = max(sources.max(), destinations.max())
    num_total_edges = len(sources)

    total_node_set = set(np.unique(np.hstack([graph_df.u.values, graph_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    
    valid_train_flag = (timestamps <= val_time)
    
    train_src_l = sources[valid_train_flag]
    train_dst_l = destinations[valid_train_flag]
    train_ts_l = timestamps[valid_train_flag]
    train_e_idx_l = edge_idxs[valid_train_flag]
    train_label_l = labels[valid_train_flag]
    train_u_index_l = u_index[valid_train_flag]
    train_i_index_l = i_index[valid_train_flag]

    valid_train_userset = set(np.unique(train_src_l))
    valid_train_itemset = set(np.unique(train_dst_l))

    u_start_idx = {}
    i_start_idx = {}

    if start_percent > 0:
        start_idx = int(train_src_l.shape[0] * start_percent)
        train_src_l = train_src_l[start_idx:]
        train_dst_l = train_dst_l[start_idx:]
        train_ts_l = train_ts_l[start_idx:]
        train_e_idx_l = train_e_idx_l[start_idx:]
        train_label_l = train_label_l[start_idx:]
        train_u_index_l = train_u_index_l[start_idx:]
        train_i_index_l = train_i_index_l[start_idx:]
    
    for (u, i, u_idx, i_idx) in zip(train_src_l, train_dst_l, train_u_index_l, train_i_index_l):
        if u not in u_start_idx:
            u_start_idx[u] = u_idx
        if i not in i_start_idx:
            i_start_idx[i] = i_idx
    for u in valid_train_userset:
        if u not in u_start_idx:
            u_start_idx[u] = 0
    for i in valid_train_itemset:
        if i not in i_start_idx:
            i_start_idx[i] = 0

    # train data -- Data
    train_data = Data(train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l, train_u_index_l, train_i_index_l, u_start_idx, i_start_idx)
    full_data.u_start_idx = u_start_idx
    full_data.i_start_idx = i_start_idx

    # valid_train_userset = set(np.unique(train_src_l))
    # valid_train_itemset = set(np.unique(train_dst_l))
    
    # select validation and test dataset
    valid_val_flag = (timestamps <= test_time) * (timestamps > val_time)
    valid_test_flag = timestamps > test_time
    # 2/8 test
    valid_test_flag_pre = (timestamps <= test_time_2) * (timestamps > test_time)
    valid_test_flag_aft = timestamps > test_time_2
    
    # validation and test with all edges
    val_src_l = sources[valid_val_flag]
    val_dst_l = destinations[valid_val_flag]
    val_ts_l = timestamps[valid_val_flag]
    val_e_idx_l = edge_idxs[valid_val_flag]
    val_label_l = labels[valid_val_flag]
    val_u_index_l = u_index[valid_val_flag]
    val_i_index_l = i_index[valid_val_flag]
    
    valid_is_old_node_edge = np.array([(a in valid_train_userset and b in valid_train_itemset) for a, b in zip(val_src_l, val_dst_l)])
    val_src_ln = val_src_l[valid_is_old_node_edge]
    val_dst_ln = val_dst_l[valid_is_old_node_edge]
    val_ts_ln = val_ts_l[valid_is_old_node_edge]
    val_e_idx_ln = val_e_idx_l[valid_is_old_node_edge]
    val_label_ln = val_label_l[valid_is_old_node_edge]
    val_u_index_ln = val_u_index_l[valid_is_old_node_edge]
    val_i_index_ln = val_i_index_l[valid_is_old_node_edge]

    # valid data -- Data
    val_data = Data(val_src_ln, val_dst_ln, val_ts_ln, val_e_idx_ln, val_label_ln, val_u_index_ln, val_i_index_ln, u_start_idx, i_start_idx)
    print('#interactions in valid: ', len(val_src_ln), flush=True)

    test_src_l = sources[valid_test_flag]
    test_dst_l = destinations[valid_test_flag]
    test_ts_l = timestamps[valid_test_flag]
    test_e_idx_l = edge_idxs[valid_test_flag]
    test_label_l = labels[valid_test_flag]
    test_u_index_l = u_index[valid_test_flag]
    test_i_index_l = i_index[valid_test_flag]
    
    test_is_old_node_edge = np.array([(a in valid_train_userset and b in valid_train_itemset) for a, b in zip(test_src_l, test_dst_l)])
    test_src_ln = test_src_l[test_is_old_node_edge]
    test_dst_ln = test_dst_l[test_is_old_node_edge]
    test_ts_ln = test_ts_l[test_is_old_node_edge]
    test_e_idx_ln = test_e_idx_l[test_is_old_node_edge]
    test_label_ln = test_label_l[test_is_old_node_edge]
    test_u_index_ln = test_u_index_l[test_is_old_node_edge]
    test_i_index_ln = test_i_index_l[test_is_old_node_edge]

    # # skip slice
    # test_src_ln = test_src_ln[0: -1: 10]
    # test_dst_ln = test_dst_ln[0: -1: 10]
    # test_ts_ln = test_ts_ln[0: -1: 10]
    # test_e_idx_ln = test_e_idx_ln[0: -1: 10]
    # test_label_ln = test_label_ln[0: -1: 10]

    # test data -- Data
    test_data = Data(test_src_ln, test_dst_ln, test_ts_ln, test_e_idx_ln, test_label_ln, test_u_index_ln, test_i_index_ln, u_start_idx, i_start_idx)
    print('#interaction in test: ', len(test_src_ln), flush=True)

    # test data pre 20%
    test_src_l_pre = sources[valid_test_flag_pre]
    test_dst_l_pre = destinations[valid_test_flag_pre]
    test_ts_l_pre = timestamps[valid_test_flag_pre]
    test_e_idx_l_pre = edge_idxs[valid_test_flag_pre]
    test_label_l_pre = labels[valid_test_flag_pre]
    test_u_index_l_pre = u_index[valid_test_flag_pre]
    test_i_index_l_pre = i_index[valid_test_flag_pre]
    
    test_is_old_node_edge_pre = np.array([(a in valid_train_userset and b in valid_train_itemset) for a, b in zip(test_src_l_pre, test_dst_l_pre)])
    test_src_ln_pre = test_src_l_pre[test_is_old_node_edge_pre]
    test_dst_ln_pre = test_dst_l_pre[test_is_old_node_edge_pre]
    test_ts_ln_pre = test_ts_l_pre[test_is_old_node_edge_pre]
    test_e_idx_ln_pre = test_e_idx_l_pre[test_is_old_node_edge_pre]
    test_label_ln_pre = test_label_l_pre[test_is_old_node_edge_pre]
    test_u_index_ln_pre = test_u_index_l_pre[test_is_old_node_edge_pre]
    test_i_index_ln_pre = test_i_index_l_pre[test_is_old_node_edge_pre]

    # test data -- Data
    test_data_pre = Data(test_src_ln_pre, test_dst_ln_pre, test_ts_ln_pre, test_e_idx_ln_pre, test_label_ln_pre, test_u_index_ln_pre, test_i_index_ln_pre)
    print('#interaction in test: ', len(test_src_ln_pre), flush=True)

    # test data pre 80%
    test_src_l_aft = sources[valid_test_flag_aft]
    test_dst_l_aft = destinations[valid_test_flag_aft]
    test_ts_l_aft = timestamps[valid_test_flag_aft]
    test_e_idx_l_aft = edge_idxs[valid_test_flag_aft]
    test_label_l_aft = labels[valid_test_flag_aft]
    test_u_index_l_aft = u_index[valid_test_flag_aft]
    test_i_index_l_aft = i_index[valid_test_flag_aft]
    
    test_is_old_node_edge_aft = np.array([(a in valid_train_userset and b in valid_train_itemset) for a, b in zip(test_src_l_aft, test_dst_l_aft)])
    test_src_ln_aft = test_src_l_aft[test_is_old_node_edge_aft]
    test_dst_ln_aft = test_dst_l_aft[test_is_old_node_edge_aft]
    test_ts_ln_aft = test_ts_l_aft[test_is_old_node_edge_aft]
    test_e_idx_ln_aft = test_e_idx_l_aft[test_is_old_node_edge_aft]
    test_label_ln_aft = test_label_l_aft[test_is_old_node_edge_aft]
    test_u_index_ln_aft = test_u_index_l_aft[test_is_old_node_edge_aft]
    test_i_index_ln_aft = test_i_index_l_aft[test_is_old_node_edge_aft]

    # test data -- Data
    test_data_aft = Data(test_src_ln_aft, test_dst_ln_aft, test_ts_ln_aft, test_e_idx_ln_aft, test_label_ln_aft, test_u_index_ln_aft, test_i_index_ln_aft)
    print('#interaction in test: ', len(test_src_ln_aft), flush=True)

    # ---------------------------------------- #

    # node_set = set(sources) | set(destinations)
    # n_total_unique_nodes = len(node_set)

    # # Compute nodes which appear at test time
    # test_node_set = set(sources[timestamps > val_time]).union(
    #     set(destinations[timestamps > val_time]))
    # # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # # their edges from training
    # new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

    # # Mask saying for each source and destination whether they are new test nodes
    # new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    # new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # # Mask which is true for edges with both destination and source not being new test nodes (because
    # # we want to remove all edges involving any new test node)
    # observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # # For train we keep edges happening before the validation time which do not involve any new node
    # # used for inductiveness
    # train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

    # train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
    #                   edge_idxs[train_mask], labels[train_mask])

    # # define the new nodes sets for testing inductiveness of the model
    # train_node_set = set(train_data.sources).union(train_data.destinations)
    # assert len(train_node_set & new_test_node_set) == 0
    # new_node_set = node_set - train_node_set

    # val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    # test_mask = timestamps > test_time

    # if different_new_nodes_between_val_and_test:
    #     n_new_nodes = len(new_test_node_set) // 2
    #     val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    #     test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    #     edge_contains_new_val_node_mask = np.array(
    #         [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    #     edge_contains_new_test_node_mask = np.array(
    #         [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
    #     new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    #     new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


    # else:
    #     edge_contains_new_node_mask = np.array(
    #         [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    #     new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    #     new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # # validation and test with all edges
    # val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
    #                 edge_idxs[val_mask], labels[val_mask])

    # test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
    #                  edge_idxs[test_mask], labels[test_mask])

    # # validation and test with edges that at least has one new node (not in training set)
    # new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
    #                          timestamps[new_node_val_mask],
    #                          edge_idxs[new_node_val_mask], labels[new_node_val_mask])

    # new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
    #                           timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
    #                           labels[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes), flush=True)
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes), flush=True)
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes), flush=True)
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes), flush=True)
    # print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    #     new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    # print("The new node test dataset has {} interactions, involving {} different nodes".format(
    #     new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    # print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    #     len(new_test_node_set)))

    return edge_features, full_data, train_data, val_data, test_data, test_data_pre, test_data_aft


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
