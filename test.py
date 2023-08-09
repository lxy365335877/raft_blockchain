from fl.utils import util
from fl.models.Update import train_cnn_mlp
import raftos
from raftos.server import Node

import argparse
import asyncio
from datetime import datetime
import random
import time

import multiprocessing

# (
#     dataset_train,
#     dataset_test,
#     dict_users,
#     test_users,
#     skew_users,
# ) = util.dataset_loader("mnist", 10, True, 2)


async def test_node(port, loop):
    raftos.configure(
        {
            "log_path": f"./node/{port}/",
            "serializer": raftos.serializers.JSONSerializer,
        }
    )
    node = Node(address=("127.0.0.1", int(port)), loop=loop)
    await node.start()

    cluster = ["127.0.0.1:{}".format(port) for port in [i + 7000 for i in range(3)]]

    for address in cluster:
        host, port = address.rsplit(":", 1)
        port = int(port)

        if (host, port) != (node.host, node.port):
            node.update_cluster((host, port))
    print(port)
    node.broadcast("broadcast")


def single_node(port):
    loop = asyncio.new_event_loop()
    loop.create_task(test_node(port, loop))
    loop.run_forever()


# for i in range(3):
#     p = multiprocessing.Process(target=single_node, args=(7000 + i,))
#     p.run()


# def test_model(id):
#     model = util.model_loader("cnn", "mnist", "cpu", 1, 10, 28)

#     while True:
#         w_local, loss = train_cnn_mlp(
#             model, dataset_train, dict_users[id], 1, "cpu", 0.01, 1
#         )
#         print(id, loss)
#         time.sleep(2)
import math

print(math.ceil(7 / 3))
