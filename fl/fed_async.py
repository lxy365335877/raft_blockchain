import logging
import os
import random
import sys
import time
import copy
import threading
import torch
from flask import Flask, request

# 这段代码实现了一个联邦学习的客户端应用，使用了Flask作为Web框架。
# 以下是代码的主要功能和结构：
# 初始化：在init()函数中加载数据集和全局模型，并初始化一些全局变量。
# 训练：train()函数用于训练客户端模型。它按照指定的轮数(args.epochs)进行训练，每轮训练完成后进行模型测试，并将结果记录在日志中。
# 启动训练：在start_train()函数中，通过threading.Thread创建一个新线程，并在该线程中调用train()函数。
# 测试：test()函数接收数据，并将其包装为一个字典，返回给调用者。
# 用户ID管理：load_user_id()函数用于加载用户ID，fetch_user_id()函数从服务器获取用户ID。
# 触发处理：my_route()函数定义了一个路由，处理来自服务器的触发事件，根据不同的消息类型(message)执行相应的动作。
# 主函数：main()函数负责解析参数，初始化全局模型和数据集，然后在新线程中启动训练，并在主线程中运行Flask应用。
# Flask应用：在__name__ == "__main__"部分，调用main()函数启动整个客户端应用，同时创建了一个Flask应用，并指定了路由处理函数。
# 该应用中的联邦学习客户端将加载数据集和全局模型，并启动一个新线程进行训练。同时，它通过Flask应用监听指定的端口，并等待来自服务器的触发事件。
# 当服务器发出触发事件时，客户端将根据不同的消息类型执行相应的操作。
# 这样，服务器可以通过发送触发事件来控制客户端的行为，比如获取用户ID、启动/停止训练等。


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
    real_path = os.path.dirname(os.path.realpath(__file__))
    peer_address_list = utils.util.env_from_sourcing(
        os.path.join(real_path, "../fabric-network/network.config"), "PeerAddress"
    ).split(" ")
    peer_header_addr = peer_address_list[0].split(":")[0]
    # initially the blockchain communicate server is load on the first peer
    blockchain_server_url = (
        "http://" + peer_header_addr + ":3000/invoke/mychannel/fabcar"
    )
    # initially the trigger url is load on the first peer
    trigger_url = "http://" + peer_header_addr + ":" + str(fed_listen_port) + "/trigger"

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


# STEP #1

# 这个`start()`函数负责在区块链账本上上传MD5哈希值，并开始联邦学习的过程。
# 它发送一个POST请求，将相关的信息上传到区块链。主要步骤如下：
# 1. 创建上传的数据体（body_data）：包含了一些重要的信息，例如消息（"Start"），
# 全局模型的MD5哈希值（global_model_hash），参与训练的用户数量（args.num_users），
# 训练轮数（args.epochs）以及是否同步（is_sync）。


# 2. 使用`http_client_post()`函数：该函数负责将数据体以POST请求的方式发送到区块链服务器（blockchain_server_url）。
# 这样，联邦学习的过程就开始了，参与的用户会开始对各自的本地数据进行训练，并将训练结果上传到区块链中，
# 最终生成一个全局模型。如果您对这个过程有任何疑问，请随时向我提问。
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


# STEP #2
# 这是第二个步骤（Step #2）的代码。这个函数负责训练本地模型，并将本地模型的更新上传到联邦学习服务器。


# 主要步骤如下：
# 1. 初始化本地模型和其他参数：首先，该函数会初始化本地模型（w_local）并记录开始训练的时间（train_start_time）。
# 2. 训练本地模型：使用`local_update`函数进行本地模型训练。这里会对数据集进行迭代，计算损失并进行反向传播，然后更新本地模型的权重。
# 3. 攻击者模拟：如果当前用户在攻击者列表中（`args.poisoning_attackers`），则会对本地模型的权重进行扰动（`utils.util.disturb_w`）。
# 4. 发送本地模型更新：将本地模型的压缩版本（`w_local_compressed`）发送给联邦学习服务器，以便进行模型聚合。
# 5. 记录本地模型哈希：将本地模型的MD5哈希值（`model_md5`）上传到区块链服务器。
# 6. 聚合全局模型：等待联邦学习服务器完成全局模型聚合（`round_finish`函数），然后继续下一轮训练。
# 请注意，这个函数的工作是在联邦学习过程的每一轮训练中为每个参与者进行本地模型的训练，并将更新上传到联邦学习服务器。
def train(uuid, epochs, start_time):
    global g_my_uuid
    global g_init_time
    logger.debug("Train local model for user: %s, epoch: %s." % (uuid, epochs))
    if g_my_uuid == -1:
        g_my_uuid = uuid  # init my_uuid at the first time

    # calculate initial model accuracy, record it as the bench mark.
    idx = int(uuid) - 1
    if epochs == args.epochs:
        # download initial global model
        body_data = {
            "message": "global_model",
        }
        logger.debug("fetch initial global model from: %s" % trigger_url)
        # time.sleep(20000)  # test to pause
        result = utils.util.http_client_post(trigger_url, body_data)
        detail = result.get("detail")
        global_model_compressed = detail.get("global_model")
        w_glob = utils.util.decompress_tensor(global_model_compressed)
        w_glob_hash = utils.util.generate_md5_hash(w_glob)
        logger.debug("Downloaded initial global model hash: " + w_glob_hash)
        net_glob.load_state_dict(w_glob)
        g_init_time[str(uuid)] = start_time
        net_glob.eval()
        (
            acc_local,
            acc_local_skew1,
            acc_local_skew2,
            acc_local_skew3,
            acc_local_skew4,
        ) = utils.util.test_model(
            net_glob, dataset_test, args, test_users, skew_users, idx
        )
        utils.util.record_log(
            uuid,
            0,
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [
                acc_local,
                acc_local_skew1,
                acc_local_skew2,
                acc_local_skew3,
                acc_local_skew4,
            ],
            args.model,
            clean=True,
        )
    train_start_time = time.time()
    w_local, _ = local_update(
        copy.deepcopy(net_glob).to(args.device), dataset_train, dict_users[idx], args
    )
    # fake attackers
    if str(uuid) in args.poisoning_attackers:
        logger.debug(
            "Detected id in poisoning attackers' list: {}, manipulate local gradients!".format(
                args.poisoning_attackers
            )
        )
        w_local = utils.util.disturb_w(w_local)
    train_time = time.time() - train_start_time

    # send local model to the first node for aggregation
    w_local_compressed = utils.util.compress_tensor(w_local)
    body_data = {
        "message": "train_ready",
        "uuid": str(uuid),
        "epochs": epochs,
        "w_compressed": w_local_compressed,
        "start_time": start_time,
        "train_time": train_time,
    }
    utils.util.http_client_post(trigger_url, body_data)

    # send hash of local model to the ledger
    model_md5 = utils.util.generate_md5_hash(w_local)
    body_data = {
        "message": "UploadLocalModel",
        "data": {
            "w": model_md5,
        },
        "uuid": uuid,
        "epochs": epochs,
        "is_sync": False,
    }
    utils.util.http_client_post(blockchain_server_url, body_data)

    # finished aggregate global model, continue next round
    round_finish(uuid, epochs)


# STEP #3
# 这是第三个步骤（Step #3）的代码。在这个步骤中，聚合服务器会接收来自参与者的本地模型更新，并根据算法（`FadeFedAvg`）聚合这些本地模型，生成新的全局模型（`w_glob`）。


# 主要步骤如下：
# 1. 接收本地模型更新：聚合服务器会接收来自参与者的本地模型更新，包括参与者的UUID、训练轮次（epochs）、训练开始时间（start_time）、
# 训练时间（train_time）、以及本地模型的压缩版本（`w_compressed`）。
# 2. 更新全局模型：聚合服务器根据接收到的本地模型更新，使用`FadeFedAvg`算法聚合全局模型（`g_train_global_model`）和本地模型（`w_local`），得到新的全局模型（`w_glob`）。
# 3. 计算并记录新的全局模型的准确率：聚合服务器会对新的全局模型进行测试，然后将准确率记录到日志中。
# 4. 更新全局模型哈希和版本：更新全局模型的压缩版本（`g_train_global_model_compressed`）、
#     哈希值（`global_model_hash`）以及版本号（`g_train_global_model_version`）。
# 5. 将全局模型哈希上传到区块链：将全局模型的哈希值上传到区块链服务器，以便其他节点下载更新的全局模型。
# 请注意，这个步骤是联邦学习过程的一个关键步骤，它负责聚合来自不同参与者的本地模型，并生成更新的全局模型。如果您有任何疑问，请随时向我提问。
def aggregate(epochs, uuid, start_time, train_time, w_compressed):
    global g_start_time
    global g_train_time
    global g_train_global_model
    global g_train_global_model_compressed
    global g_train_global_model_version
    global global_model_hash

    logger.debug("Received a train_ready.")
    lock.acquire()
    key = str(uuid) + "-" + str(epochs)
    g_start_time[key] = start_time
    g_train_time[key] = train_time
    lock.release()
    # mimic DDoS attacks here
    if args.ddos_duration == 0 or args.ddos_duration > g_train_global_model_version:
        logger.debug("Mimic the aggregator under DDoS attacks!")
        if random.random() < args.ddos_no_response_percent:
            logger.debug(
                "Unfortunately, the aggregator does not response to the local update gradients"
            )
            lock.acquire()
            # ignore the update to the global model
            g_train_global_model_version += 1
            lock.release()
            return

    logger.debug("Aggregate global model after received a new local model.")
    w_local = utils.util.decompress_tensor(w_compressed)
    # aggregate global model
    if g_train_global_model is not None:
        fade_c = calculate_fade_c(
            uuid, w_local, args.fade, args.model, args.poisoning_detect_threshold
        )
        w_glob = FadeFedAvg(g_train_global_model, w_local, fade_c)
    # test new global model acc and record onto the log
    intermediate_acc_record(w_glob)
    # save global model for further download
    g_train_global_model_compressed = utils.util.compress_tensor(w_glob)
    lock.acquire()
    g_train_global_model = w_glob
    g_train_global_model_version += 1
    lock.release()
    # generate hash of global model
    global_model_hash = utils.util.generate_md5_hash(w_glob)
    logger.debug(
        "As a committee leader, calculate new global model hash: " + global_model_hash
    )
    # send the download link and hash of global model to the ledger
    body_data = {
        "message": "UploadGlobalModel",
        "data": {
            "global_model_hash": global_model_hash,
        },
        "uuid": uuid,
        "epochs": epochs,
        "is_sync": False,
    }
    logger.debug(
        "aggregate global model finished, send global_model_hash [%s] to blockchain in epoch [%s]."
        % (global_model_hash, epochs)
    )
    utils.util.http_client_post(blockchain_server_url, body_data)


# `calculate_fade_c` 函数用于计算 `FadeFedAvg` 算法中的聚合系数 `fade_c`，该系数用于加权平均本地模型和全局模型。在计算 `fade_c` 时，
# 根据选择的动态设置或静态设置来调整该系数。
# 函数的主要步骤如下
# 1. 动态设置或静态设置 `fade_c`：如果 `fade_target` 为 -1，表示采用动态设置，
# 需要先测试新的本地模型在验证集上的准确率。如果 `fade_target` 不为 -1，表示采用静态设置，直接使用设定的值作为 `fade_c`。
# 2. 测试新的本地模型准确率：将本地模型加载到全局模型中，测试其在验证集上的准确率，得到 `acc_local`。
# 3. 计算 `fade_c` 值：根据模型类型（"lstm"、"cnn" 或 "mlp"）以及当前的 `acc_local` 和之前的
# `current_acc_local`，计算 `fade_c` 的值。
# 4. 检查 `fade_c` 是否小于阈值：如果计算得到的 `fade_c` 值小于 `acc_detect_threshold`，则将
# `fade_c` 设为 0，以过滤掉可能是恶意攻击的本地更新。
# 5. 返回计算得到的 `fade_c` 值。


# 总的来说，这个函数负责计算 `FadeFedAvg` 算法中的聚合系数 `fade_c`，确保在动态设置时考虑本地模型的准确率，并根据需要过滤掉可能是恶意攻击的本地更新。
def calculate_fade_c(uuid, w_local, fade_target, model, acc_detect_threshold):
    if fade_target == -1:  # -1 means fade dynamic setting
        logger.debug("fade=-1, dynamic fade setting is adopted!")
        # dynamic fade setting, test new acc_local first
        net_glob.load_state_dict(w_local)
        net_glob.eval()
        idx = int(uuid) - 1

        (
            acc_local,
            acc_local_skew1,
            acc_local_skew2,
            acc_local_skew3,
            acc_local_skew4,
        ) = utils.util.test_model(
            net_glob, dataset_test, args, test_users, skew_users, idx
        )
        logger.debug(
            "after test, acc_local: {}, current_acc_local: {}".format(
                acc_local, current_acc_local
            )
        )
        if model == "lstm":
            # for lstm, acc_local means the mse loss instead of accuracy, the less the better
            if current_acc_local == -1:
                fade_c = 10
            else:
                try:
                    fade_c = current_acc_local / acc_local
                except ZeroDivisionError as err:
                    logger.debug(
                        "Divided by zero: {}, set scaling factor to 10 by default.".format(
                            err
                        )
                    )
                    fade_c = 10
        else:
            # for cnn or mlp models, accuracy the higher the better.
            if current_acc_local == -1:
                fade_c = 10
            else:
                try:
                    fade_c = acc_local / current_acc_local
                except ZeroDivisionError as err:
                    logger.debug(
                        "Divided by zero: {}, set scaling factor to 10 by default.".format(
                            err
                        )
                    )
                    fade_c = 10
        # filter out poisoning local updated gradients whose test accuracy is less than acc_detect_threshold
        if fade_c < acc_detect_threshold:
            fade_c = 0
    else:
        logger.debug("fade={}, static fade setting is adopted!".format(fade_target))
        # static fade setting
        fade_c = fade_target
    logger.debug("calculated fade_c: %f" % fade_c)
    return fade_c


# `intermediate_acc_record` 函数用于在每一轮聚合结束后，记录当前节点的模型在验证集上的准确率，并将相关信息记录到日志中。


# 函数的主要步骤如下：
# 1. 加载全局模型：将参数 `w_glob` 加载到全局模型 `net_glob` 中。
# 2. 设置全局模型为评估模式：将全局模型设置为评估模式，即 `net_glob.eval()`，以便在验证集上进行准确率测试。
# 3. 计算时间和准确率：通过调用 `utils.util.test_model` 函数计算当前节点的全局模型在验证集上的准确率。`test_model`
# 函数将返回本地模型在验证集上的准确率 `acc_local`，
# 以及针对特定类别样本进行的四个子准确率 `acc_local_skew1`、`acc_local_skew2`、`acc_local_skew3` 和 `acc_local_skew4`。
# 4. 记录日志：调用 `utils.util.record_log` 函数将节点的信息记录到日志中。
# 日志包含当前节点的ID `g_my_uuid`、时间 `total_time`、准确率信息 `acc_local`、
# `acc_local_skew1`、`acc_local_skew2`、`acc_local_skew3` 和 `acc_local_skew4`，以及所采用的模型类型 `args.model`。
# 总的来说，这个函数负责在每一轮聚合结束后，记录当前节点模型在验证集上的准确率，并将相关信息记录到日志中，以便后续分析和监控模型训练的进度和效果。
def intermediate_acc_record(w_glob):
    net_glob.load_state_dict(w_glob)
    net_glob.eval()
    total_time = time.time() - g_init_time[str(g_my_uuid)]
    idx = int(g_my_uuid) - 1
    (
        acc_local,
        acc_local_skew1,
        acc_local_skew2,
        acc_local_skew3,
        acc_local_skew4,
    ) = utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)
    utils.util.record_log(
        g_my_uuid,
        0,
        [total_time, 0.0, 0.0, 0.0, 0.0],
        [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4],
        args.model,
    )


# STEP #7

# `round_finish` 函数用于在每一轮聚合完成后执行一系列操作，
# 包括下载最新的全局模型，测试节点的本地模型在验证集上的准确率，并记录训练和测试时间等信息到日志中。
# 函数的主要步骤如下：
# 1. 下载最新的全局模型：通过向链码服务发送请求，下载最新的全局模型的压缩版本，并将其解压缩得到 `w_glob`。
# 2. 更新全局模型的哈希值：计算新的全局模型 `w_glob` 的哈希值，并将其更新为 `global_model_hash`。
# 3. 倒计时轮次：将轮次 `epochs` 减去1，得到 `new_epochs`。
# 4. 获取训练时间和测试准确率：通过向链码服务发送请求，获取训练开始时间 `start_time` 和训练时间 `train_time`。
# 5. 测试本地模型准确率：将全局模型 `w_glob` 加载到 `net_glob` 中，并设置其为评估模式。然后使用 `utils.util.test_model`
# 函数计算当前节点在验证集上的准确率 `acc_local`，以及针对特定类别样本进行的四个子准确率 `acc_local_skew1`、
# `acc_local_skew2`、`acc_local_skew3` 和 `acc_local_skew4`。
# 6. 记录日志：调用 `utils.util.record_log` 函数将节点的信息记录到日志中。日志包含节点的ID `uuid`、轮次 `epochs`、
# 总时间 `total_time`、本轮时间 `round_time`、训练时间 `train_time`、测试时间 `test_time` 和
# 通信时间 `communication_time`，以及本轮测试的准确率信息 `acc_local`、`acc_local_skew1`、`acc_local_skew2`、
# `acc_local_skew3` 和 `acc_local_skew4`，以及所采用的模型类型 `args.model`。
# 7. 判断是否开始下一轮训练：如果 `new_epochs` 大于0，则开始下一轮训练，调用 `train` 函数。
# 否则，输出 "ALL DONE!" 的提示信息，并向链码服务发送 "shutdown_python" 的请求，以终止当前节点的运行。


# 总的来说，这个函数负责在每一轮聚合完成后，执行一系列操作，包括下载最新的全局模型、测试本地模型的准确率，并记录相关信息到日志中。
# 然后判断是否继续进行下一轮训练，或者结束节点的运行。
def round_finish(uuid, epochs):
    global global_model_hash
    global current_acc_local
    logger.debug(
        "Download latest global model for user: %s, epoch: %s." % (uuid, epochs)
    )

    # download global model
    body_data = {
        "message": "global_model",
    }
    result = utils.util.http_client_post(trigger_url, body_data)
    detail = result.get("detail")
    global_model_compressed = detail.get("global_model")
    global_model_version = detail.get("version")
    logger.debug(
        "Successfully fetched global model [%s] of epoch [%s] from: %s"
        % (global_model_version, epochs, trigger_url)
    )
    w_glob = utils.util.decompress_tensor(global_model_compressed)
    # load hash of new global model, which is downloaded from the leader
    global_model_hash = utils.util.generate_md5_hash(w_glob)
    logger.debug("Received new global model with hash: " + global_model_hash)

    # epochs count backwards until 0
    new_epochs = epochs - 1
    # fetch time record
    fetch_data = {
        "message": "fetch_time",
        "uuid": uuid,
        "epochs": epochs,
    }
    response = utils.util.http_client_post(trigger_url, fetch_data)
    detail = response.get("detail")
    start_time = detail.get("start_time")
    train_time = detail.get("train_time")

    # finally, test the acc_local, acc_local_skew1~4
    net_glob.load_state_dict(w_glob)
    net_glob.eval()
    test_start_time = time.time()
    idx = int(uuid) - 1
    (
        acc_local,
        acc_local_skew1,
        acc_local_skew2,
        acc_local_skew3,
        acc_local_skew4,
    ) = utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)
    current_acc_local = acc_local
    test_time = time.time() - test_start_time

    # before start next round, record the time
    total_time = time.time() - g_init_time[str(uuid)]
    round_time = time.time() - start_time
    communication_time = utils.util.reset_communication_time()
    utils.util.record_log(
        uuid,
        epochs,
        [total_time, round_time, train_time, test_time, communication_time],
        [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4],
        args.model,
    )
    if new_epochs > 0:
        # start next round of train right now
        train(uuid, new_epochs, time.time())
    else:
        logger.info("########## ALL DONE! ##########")
        body_data = {"message": "shutdown_python"}
        utils.util.http_client_post(trigger_url, body_data)


# 这三个函数在代码中用于不同的目的：

# 1. `shutdown_count()`: 用于统计已经调用了 `shutdown_count()` 的次数，一旦达到了所有节点的数量 `args.num_users`，
# 则向区块链服务发送 "ShutdownPython" 请求，以通知区块链服务终止所有节点的运行。

# 2. `fetch_time(uuid, epochs)`: 用于返回指定节点和轮次的训练开始时间 `start_time` 和训练时间 `train_time`。
# 在 `round_finish` 函数中被调用，用于获取其他节点的训练时间信息。


# 3. `download_global_model()`: 用于返回当前全局模型的压缩版本 `g_train_global_model_compressed` 和
# 全局模型版本号 `g_train_global_model_version`。在 `aggregate` 函数中被调用，用于将最新的全局模型发送给其他节点。
# 总的来说，这些函数的功能是：
# - `shutdown_count()`：用于统计已经调用了 `shutdown_count()` 的次数，当所有节点都调用了该函数后，向区块链服务发送终止节点的请求。
# - `fetch_time(uuid, epochs)`：用于获取指定节点和轮次的训练开始时间和训练时间。
# - `download_global_model()`：用于获取当前全局模型的压缩版本和全局模型版本号，以备在聚合时发送给其他节点。
def shutdown_count():
    global shutdown_count_num
    lock.acquire()
    shutdown_count_num += 1
    lock.release()
    if shutdown_count_num == args.num_users:
        # send request to blockchain for shutting down the python
        body_data = {
            "message": "ShutdownPython",
            "data": {},
            "uuid": "",
            "epochs": 0,
            "is_sync": False,
        }
        logger.debug("Sent shutdown python request to blockchain.")
        utils.util.http_client_post(blockchain_server_url, body_data)


def fetch_time(uuid, epochs):
    key = str(uuid) + "-" + str(epochs)
    start_time = g_start_time.get(key)
    train_time = g_train_time.get(key)
    detail = {
        "start_time": start_time,
        "train_time": train_time,
    }
    return detail


def download_global_model():
    detail = {
        "global_model": g_train_global_model_compressed,
        "version": g_train_global_model_version,
    }
    return detail


# 这是一个Flask应用中的路由处理函数定义。
# Flask应用是一个Web应用程序框架，通过定义不同的路由处理函数来处理不同的URL请求。
# 在这个代码段中，定义了三个路由处理函数：
# 1. `main_handler()`: 处理URL路径 '/messages' 的请求。对于GET请求，调用 `start()` 函数，然后返回一个简单的JSON响应表示成功。
# 对于POST请求，解析请求中的JSON数据，根据数据中的 "message" 字段选择执行相应的处理函数，比如调用 `train()` 或 `utils.util.my_exit()` 函数。
# 2. `trigger_handler()`: 处理URL路径 '/trigger' 的请求。
# 对于POST请求，解析请求中的JSON数据，根据数据中的 "message" 字段选择执行相应的处理函数，
# 比如调用 `aggregate()`、`download_global_model()` 或 `fetch_time()` 函数，并返回相应的JSON响应。
# 3. `test()`: 处理URL路径 '/test' 的请求。对于GET请求，返回一个包含 "test": "success" 的JSON响应。
# 对于POST请求，解析请求中的JSON数据，并原样返回解析得到的JSON数据。
# 这些路由处理函数定义了应用中不同URL请求的处理方式，根据不同的URL路径和请求类型，选择调用相应的功能函数并返回相应的响应。
def my_route(app):
    @app.route("/messages", methods=["GET", "POST"])
    def main_handler():
        # For GET
        if request.method == "GET":
            start()
            response = {"status": "yes"}
            return response
        # For POST
        else:
            data = request.get_json()
            status = "yes"
            detail = {}
            response = {"status": status, "detail": detail}
            # Then judge message type and process
            message = data.get("message")
            if message == "prepare":
                threading.Thread(
                    target=train,
                    args=(data.get("uuid"), data.get("epochs"), time.time()),
                ).start()
            elif message == "shutdown":
                threading.Thread(
                    target=utils.util.my_exit, args=(args.exit_sleep,)
                ).start()
            return response

    @app.route("/trigger", methods=["GET", "POST"])
    def trigger_handler():
        # For POST
        if request.method == "POST":
            data = request.get_json()
            status = "yes"
            detail = {}
            message = data.get("message")
            if message == "train_ready":
                threading.Thread(
                    target=aggregate,
                    args=(
                        data.get("epochs"),
                        data.get("uuid"),
                        data.get("start_time"),
                        data.get("train_time"),
                        data.get("w_compressed"),
                    ),
                ).start()
            elif message == "global_model":
                detail = download_global_model()
            elif message == "fetch_time":
                detail = fetch_time(data.get("uuid"), data.get("epochs"))
            elif message == "shutdown_python":
                threading.Thread(target=shutdown_count, args=()).start()
            response = {"status": status, "detail": detail}
            return response

    @app.route("/test", methods=["GET", "POST"])
    def test():
        # For GET
        if request.method == "GET":
            test_body = {"test": "success"}
            return test_body
        # For POST
        else:
            doc = request.get_json()
            return doc


# 在这个代码段中，是应用的主入口，即应用的启动部分。
# 1. `init()`: 初始化应用所需的全局变量，包括解析配置文件、准备数据集、模型等。
# 2. 创建一个 `Flask` 应用实例 `flask_app`。
# 3. 调用 `my_route(flask_app)`，将定义的路由处理函数绑定到 `flask_app` 实例，这样应用就能够根据不同的URL请求调用相应的处理函数。
# 4. 日志记录一条信息，表示应用正在监听指定的端口。
# 5. 最后，调用 `flask_app.run()` 启动应用，让应用开始监听并处理来自客户端的请求。
# `host='0.0.0.0'` 表示应用将在所有可用的网络接口上监听请求，`port=fed_listen_port` 表示指定的端口号。
# 这样，当运行这个Python脚本时，Flask应用会被启动，并开始监听在指定的端口上。
# 当有客户端发起请求时，应用会根据请求的URL路径和方法，调用相应的路由处理函数来处理请求，并返回相应的响应。
if __name__ == "__main__":
    init()
    flask_app = Flask(__name__)
    my_route(flask_app)
    logger.info("start serving at " + str(fed_listen_port) + "...")
    flask_app.run(host="172.18.197.104", port=fed_listen_port)
