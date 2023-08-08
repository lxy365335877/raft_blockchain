from fl.utils import util
from fl.models.Update import train_cnn_mlp, DatasetSplit, train_new
from torch.utils.data import DataLoader, Dataset
import raftos

import argparse
import asyncio
import random
import time

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


# single node process
async def run(user_idx, ldr_train):
    model = util.model_loader("cnn", "mnist", "cpu", 1, 10, 28)

    while True:
        # We can also check if raftos.get_leader() == node_id
        await raftos.wait_until_leader(f"127.0.0.1:{user_idx+base_port}")
        await asyncio.sleep(2)
        # time.sleep(2)
        # w_local, loss = train_new(model, ldr_train, 1, "cpu", 0.001, local_bs)
        # print(user_idx, loss)
        print(user_idx)


# run a single node
def single_node(cluster, user_idx, ldr_train):
    raftos.configure(
        {
            "log_path": f"./node/{user_idx}/",
            "serializer": raftos.serializers.JSONSerializer,
        }
    )
    loop = asyncio.get_event_loop()
    loop.create_task(
        raftos.register(f"127.0.0.1:{user_idx+base_port}", cluster=cluster)
    )
    loop.run_until_complete(run(user_idx, ldr_train))


if __name__ == "__main__":
    # cluster
    cluster = [
        "127.0.0.1:{}".format(port)
        for port in [i + base_port for i in range(num_users)]
    ]

    print(cluster)

    for user_idx in range(num_users):
        # ldr_train = DataLoader(
        #     DatasetSplit(dataset_train, dict_users[user_idx]),
        #     batch_size=local_bs,
        #     shuffle=True,
        # )

        p = multiprocessing.Process(
            target=single_node,
            args=(cluster, user_idx, None),
        )
        p.start()
        pass
