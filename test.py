from fl.utils import util
from fl.models.Update import train_cnn_mlp
import raftos

import argparse
import asyncio
from datetime import datetime
import random
import time

import multiprocessing

(
    dataset_train,
    dataset_test,
    dict_users,
    test_users,
    skew_users,
) = util.dataset_loader("mnist", 10, True, 2)


def test_model(id):
    model = util.model_loader("cnn", "mnist", "cpu", 1, 10, 28)

    while True:
        w_local, loss = train_cnn_mlp(
            model, dataset_train, dict_users[id], 1, "cpu", 0.01, 1
        )
        print(id, loss)
        time.sleep(2)


for i in range(2):
    p = multiprocessing.Process(target=test_model, args=(i,))
    p.start()
