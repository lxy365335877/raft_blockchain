import logging
import os
import random
import sys
import time
import copy
import threading
import torch
from flask import Flask, request

import utils.util
from utils.options import args_parser
from utils.util import dataset_loader, model_loader, ColoredLogger
from fl.models.Update import local_update
from fl.models.Fed import FadeFedAvg

logging.setLoggerClass(ColoredLogger)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger("fed_async")

# TO BE CHANGED
# federated learning server listen port
fed_listen_port = 8888
# TO BE CHANGED FINISHED

#
# 这段代码似乎是一个Federated Learning（联邦学习）的实现，定义了一些全局的变量用于控制联邦学习的过程。
# # NOT TO TOUCH VARIABLES BELOW
blockchain_server_url = ""
trigger_url = ""
args = None
net_glob = None
dataset_train = None
dataset_test = None
dict_users = []
lock = threading.Lock()
test_users = []
skew_users = []
peer_address_list = []
global_model_hash = ""
g_my_uuid = -1
g_init_time = {}
g_start_time = {}
g_train_time = {}
g_train_global_model = None
g_train_global_model_compressed = None
g_train_global_model_version = 0
shutdown_count_num = 0
current_acc_local = -1


# 这段代码是初始化Federated Learning（联邦学习）的过程。它设置了全局变量并解析配置文件中的参数。以下是这个初始化过程中涉及到的主要步骤：
# 1. 解析配置文件：从网络配置文件中获取Peer节点的地址，并根据第一个Peer的地址设置区块链服务器和触发URL的初始值。
# 2. 解析命令行参数：根据命令行参数解析设置设备类型、模型、数据集、通道数、类别数等参数。
# 3. 加载数据集和模型：根据数据集和模型名称加载相应的数据集和模型。
# 4. 设置初始全局模型：将初始训练好的本地模型作为第一轮的全局模型，并计算其MD5哈希值，以备后续使用。


# STEP #1
def init():
    global args
    global net_glob
    global dataset_train
    global dataset_test
    global dict_users
    global test_users
    global skew_users
    global blockchain_server_url
    global trigger_url
    global peer_address_list
    global global_model_hash
    global g_train_global_model
    global g_train_global_model_compressed
    # parse network.config and read the peer addresses
    # real_path = os.path.dirname(os.path.realpath(__file__))
    # peer_address_list = utils.util.env_from_sourcing(os.path.join(real_path, "../fabric-network/network.config"),
    #                                                  "PeerAddress").split(' ')
    # peer_header_addr = peer_address_list[0].split(":")[0]
    # # initially the blockchain communicate server is load on the first peer
    # blockchain_server_url = "http://" + peer_header_addr + ":3000/invoke/mychannel/fabcar"
    # # initially the trigger url is load on the first peer
    # trigger_url = "http://" + peer_header_addr + ":" + str(fed_listen_port) + "/trigger"

    # parse args
    args = args_parser()
    args.device = torch.device(
        "cuda:{}".format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else "cpu"
    )
    logger.setLevel(args.log_level)
    # parse participant number
    args.num_users = len(peer_address_list)

    dataset_train, dataset_test, dict_users, test_users, skew_users = dataset_loader(
        args.dataset, args.dataset_train_size, args.iid, args.num_users
    )
    if dataset_train is None:
        logger.error("Error: unrecognized dataset")
        sys.exit()

    img_size = dataset_train[0][0].shape
    net_glob = model_loader(
        args.model,
        args.dataset,
        args.device,
        args.num_channels,
        args.num_classes,
        img_size,
    )
    if net_glob is None:
        logger.error("Error: unrecognized model")
        sys.exit()
    # finally trained the initial local model, which will be treated as first global model.
    net_glob.train()
    # generate md5 hash from model, which is treated as global model of previous round.
    w = net_glob.state_dict()
    global_model_hash = utils.util.generate_md5_hash(w)
    g_train_global_model = w
    g_train_global_model_compressed = utils.util.compress_tensor(w)


def start():
    # upload md5 hash to ledger
    body_data = {
        "message": "Start",
        "data": {
            "global_model_hash": global_model_hash,
            "user_number": args.num_users,
        },
        "epochs": args.epochs,
        "is_sync": False,
    }
    utils.util.http_client_post(blockchain_server_url, body_data)


if __name__ == "__main__":
    init()
