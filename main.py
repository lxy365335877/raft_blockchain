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

(
    dataset_train,
    dataset_test,
    dict_users,
    test_users,
    skew_users,
) = util.dataset_loader("mnist", 10, True, num_users)


# run a single node
async def single_node(cluster, user_idx, ldr_train):
    raftos.configure(
        {
            "log_path": f"./node/{user_idx}/",
            "serializer": raftos.serializers.JSONSerializer,
        }
    )
    node_id = f"127.0.0.1:{user_idx+base_port}"
    node = await raftos.register(node_id, cluster=cluster)

    model = util.model_loader("cnn", "mnist", f"cpu", 1, 10, 28)
    data_id = raftos.Replicated(name="data_id")

    nonce = 0

    while True:
        # We can also check if raftos.get_leader() == node_id
        await raftos.wait_until_leader(node_id)
        await asyncio.sleep(2)

        w_local, loss = train_new(
            model, ldr_train, 1, f"cpu:{user_idx}", 0.001, local_bs
        )

        w_local_serialized = pickle.dumps(w_local)
        compressed_data = gzip.compress(w_local_serialized)
        b64_encoded = base64.b64encode(compressed_data)
        w_local_str = b64_encoded.decode("ascii")

        print(user_idx, loss)
        # await data_id.set(loss)
        w_local_str_split = [
            w_local_str[:48000],
            w_local_str[48000:96000],
            w_local_str[96000:],
        ]
        for i in range(len(w_local_str_split)):
            data = {
                "type": "validate",
                "term": node.state.storage.term,
                "weight_serial": nonce,
                "weight_split_no": i,
                "weight_split": w_local_str_split[i],
            }
            node.broadcast(data)
        nonce += 1


def run(cluster, user_idx, ldr_train):
    loop = asyncio.get_event_loop()
    loop.create_task(single_node(cluster, user_idx, ldr_train))
    loop.run_forever()


if __name__ == "__main__":
    # cluster
    cluster = [
        "127.0.0.1:{}".format(port)
        for port in [i + base_port for i in range(num_users)]
    ]

    print(cluster)

    for user_idx in range(num_users):
        ldr_train = DataLoader(
            DatasetSplit(dataset_train, dict_users[user_idx]),
            batch_size=local_bs,
            shuffle=True,
        )

        p = multiprocessing.Process(
            target=run,
            args=(cluster, user_idx, ldr_train),
        )
        p.start()
        pass
