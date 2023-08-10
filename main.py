from fl.utils import util
from fl.models.Update import train_cnn_mlp, DatasetSplit, train_new
from torch.utils.data import DataLoader, Dataset
import raftos

import argparse
import asyncio
import random
import time
import math

from datetime import datetime

import pickle
import base64
import gzip

import multiprocessing

num_users = 3
base_port = 7000
local_bs = 10


# run a single node
async def single_node(cluster, user_idx):
    raftos.configure(
        {
            "log_path": f"./node/{user_idx}/",
            "serializer": raftos.serializers.JSONSerializer,
        }
    )
    node_id = f"127.0.0.1:{user_idx+base_port}"
    node = await raftos.register(node_id, cluster=cluster)
    # data_id = raftos.Replicated(name="data_id")

    while True:
        # We can also check if raftos.get_leader() == node_id
        await raftos.wait_until_leader(node_id)
        await asyncio.sleep(2)

        # local train
        node.train_broadcast_w()


def run(cluster, user_idx):
    loop = asyncio.get_event_loop()
    loop.create_task(single_node(cluster, user_idx))
    loop.run_forever()


if __name__ == "__main__":
    # cluster
    cluster = [
        "127.0.0.1:{}".format(port)
        for port in [i + base_port for i in range(num_users)]
    ]

    print(cluster)

    for user_idx in range(num_users):
        p = multiprocessing.Process(
            target=run,
            args=(cluster, user_idx),
        )
        p.start()
        pass
