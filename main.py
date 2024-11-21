# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset
from models.wrapper import WrapperModel
from models.GDN import GDN

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

from models.MSTCN import MultiScale_TemporalConv, TCN1d
from torchsummary import summary
from torchstat import stat

class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None
        self.MSConv = None
        
        dataset = self.env_config['dataset']

        train, test = self.dataPreprocess(dataset)

    ### GDN部分修改，加上建图部分
        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())


        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)
        
        if train_config['use_tcn']:
            self.MSConv = TCN1d(
                feature_num=len(feature_map),
                # kernel_size=5,
                # dilation=2
            )
            # self.MSConv = MultiScaleTCN1D(
            #         feature_num=len(feature_map),
            #         kernels=[2, 3, 5, 7]
            #     ).to(self.device)

        if train_config['model'] == 'GDN':
            self.model = GDN(edge_index_sets, len(feature_map), 
                    dim=train_config['dim'], 
                    input_dim=train_config['slide_win'],
                    out_layer_num=train_config['out_layer_num'],
                    out_layer_inter_dim=train_config['out_layer_inter_dim'],
                    topk=train_config['topk'],
                    MSConv=self.MSConv,
                    use_topo=train_config['use_topo']
                ).to(self.device)
        
        elif train_config['model'] == 'AT':
            pass   

    def run(self):
        # wrapperModel = WrapperModel(self.model).to(self.device)
        # stat(wrapperModel, (1, 38, 100))
        # summary(self.model, (38, 100), batch_size=2)

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        self.get_score(self.test_result, self.val_result)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)


        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths
    

    def dataPreprocess(self, dataset):
        train_data_path_list = []
        test_data_path_list = []
        label_data_path_list = []

        # 标记数据集是否为pkl
        pkl = True

        if dataset == 'wadi' or dataset == 'swat' or dataset == 'msl':
            pkl = False
            train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
            test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)

        elif dataset == "smd":
            data_set_number = ["3-4",'3-5',"3-10","3-11","1-5","1-8","2-4"]
            data_set_number += ["1-1","1-2","1-3","1-4","1-5","1-6","1-7","1-8"]
            data_set_number += ["2-1","2-2","2-3","2-4","2-5","2-6","2-7","2-8","2-9"]
            data_set_number += ["3-1","3-2","3-3","3-4","3-5","3-6","3-7","3-8","3-9","3-10","3-11"]

            for data_set_id in data_set_number:
                file = f"machine-{data_set_id}_train.pkl"
                train_data_path_list.append("data/Machine/" + file)
                test_data_path_list.append("data/Machine/" + file.replace("_train.pkl","_test.pkl"))
                label_data_path_list.append("data/Machine/" + file.replace("_train.pkl","_test_label.pkl"))

        elif dataset == "gcp":
            data_set_number = [f"service{i}" for i in range(0,30)]
            for data_set_id in data_set_number:
                file = f"{data_set_id}_train.pkl"
                train_data_path_list.append("data/Machine/" + file)
                test_data_path_list.append("data/Machine/" + file.replace("_train.pkl","_test.pkl"))
                label_data_path_list.append("data/Machine/" + file.replace("_train.pkl","_test_label.pkl"))

        else: # for dataset with only one subset
            data_set_number = [dataset]
            for data_set_id in data_set_number:
                file = f"{data_set_id}_train.pkl"
                train_data_path_list.append("data/Machine/" + file)
                test_data_path_list.append("data/Machine/" + file.replace("_train.pkl", "_test.pkl"))
                label_data_path_list.append("data/Machine/" + file.replace("_train.pkl", "_test_label.pkl"))

        if pkl:
            # 把subset拼接起来，把test_data和label拼接起来，test_data的最后一个维度是label
            for i in range(len(train_data_path_list)):
                train_orig = pd.read_pickle(train_data_path_list[i])
                test_orig = pd.read_pickle(test_data_path_list[i])
                label_orig = pd.read_pickle(label_data_path_list[i])

                if i == 0:
                    train = train_orig
                    test = test_orig
                    label = label_orig    
                else:
                    train = np.concatenate((train, train_orig), axis=0)
                    test = np.concatenate((test, test_orig), axis=0)
                    label = np.concatenate((label, label_orig), axis=0)
                break
                    
            test = np.column_stack((test, label))
            # 为train生成列名
            train_columns = [f'{i}' for i in range(0, train.shape[1])]

            # 为test生成列名，注意最后一列是label
            test_columns = [f'{i}' for i in range(0, test.shape[1] - 1)] + ['attack']

            # 将ndarray转换为DataFrame，并指定列名
            # 除了最后一列，其余乘以20
            # 假设 train 和 test 已经被定义，并且 train_columns 和 test_columns 也已提供
            train = pd.DataFrame(train, columns=train_columns)
            test = pd.DataFrame(test, columns=test_columns)
            
            # pkl向csv数据转化
            # train.to_csv(f'./data/{dataset}/train.csv', index=False)
            # test.to_csv(f'./data/{dataset}/test.csv', index=False)
            # print("已转化完成")

            # # 扩展数据
            # train *= 20
            # test.iloc[:, :-1] *= 20

            # 初始化MinMaxScaler
            scaler = MinMaxScaler()

            # 归一化 train 数据
            train_scaled = scaler.fit_transform(train)
            train = pd.DataFrame(train_scaled, columns=train_columns)

            # 归一化 test 数据（除了最后一列标签）
            test_scaled = scaler.transform(test.iloc[:, :-1])
            # 将归一化后的数据与最后一列标签合并
            test = pd.concat([pd.DataFrame(test_scaled, columns=test_columns[:-1]), test.iloc[:, -1].reset_index(drop=True)], axis=1)

            if 'attack' in train.columns: 
                train = train.drop(columns=['attack'])
        
        else:
            train, test = train_orig, test_orig
            if 'attack' in train.columns: 
                train = train.drop(columns=['attack'])
            # 初始化MinMaxScaler
            scaler = MinMaxScaler()

            # 归一化 train 数据
            train_scaled = scaler.fit_transform(train)

            # 归一化 test 数据（除了最后一列标签）
            test_scaled = scaler.transform(test.iloc[:, :-1])
            test_scaled = pd.DataFrame(test_scaled, columns=test.columns[:-1])
            
            # 将归一化后的数据与最后一列标签合并
            test = pd.concat([test_scaled, test.iloc[:, -1].reset_index(drop=True)], axis=1)
            # test = pd.concat([test_scaled, test.iloc[:, -1].reset_index(drop=True)], axis=1)
        
        print(test.head())
        return train, test


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=16)
    parser.add_argument('-epoch', help='train epoch', type = int, default=150)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=100)
    parser.add_argument('-dim', help='dimension', type = int, default=128)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=10)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi/swat/PSM/smd/gcp/msl/SMAP', type = str, default='smd')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cpu')
    parser.add_argument('-random_seed', help='random seed', type = int, default=5)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=64)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.02)
    parser.add_argument('-topk', help='topk num', type = int, default=15)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')
    parser.add_argument('-model', help='GDN/AT', type = str, default='GDN')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'use_tcn': False,
        'use_topo': False, 
        'model': args.model
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()

