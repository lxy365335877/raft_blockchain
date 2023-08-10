import asyncio
import functools
import pickle
import gzip
import base64

from .network import UDPProtocol
from .state import State
from fl.utils import util
from fl.models.Update import DatasetSplit, train_new
from torch.utils.data import DataLoader, Dataset


async def register_bak(*address_list, cluster=None, loop=None):
    """Start Raft node (server)
    Args:
        address_list — 127.0.0.1:8000 [, 127.0.0.1:8001 ...]
        cluster — [127.0.0.1:8001, 127.0.0.1:8002, ...]
    """

    loop = loop or asyncio.get_event_loop()
    for address in address_list:
        host, port = address.rsplit(":", 1)
        node = Node(address=(host, int(port)), loop=loop)
        await node.start()

        for address in cluster:
            host, port = address.rsplit(":", 1)
            port = int(port)

            if (host, port) != (node.host, node.port):
                node.update_cluster((host, port))


async def register(address, cluster=None, loop=None):
    """Start Raft node (server)
    Args:
        address_list — 127.0.0.1:8000 [, 127.0.0.1:8001 ...]
        cluster — [127.0.0.1:8001, 127.0.0.1:8002, ...]
    """

    loop = loop or asyncio.get_event_loop()
    host, port = address.rsplit(":", 1)
    node = Node(address=(host, int(port)), loop=loop)
    await node.start()

    for address in cluster:
        host, port = address.rsplit(":", 1)
        port = int(port)

        if (host, port) != (node.host, node.port):
            node.update_cluster((host, port))

    node.init_model()
    return node


def stop():
    for node in Node.nodes:
        node.stop()


class Node:
    """Raft Node (Server)"""

    nodes = []

    def __init__(self, address, loop):
        self.host, self.port = address
        self.cluster = set()

        self.loop = loop
        self.state = State(self)
        self.requests = asyncio.Queue(loop=self.loop)
        self.__class__.nodes.append(self)

    # a raft based FL node init deeplearning model.
    def init_model(
        self,
        train_count=100,
        model_name="cnn",
        dataset_name="mnist",
        device="cpu",
        num_channels=1,
        num_classes=10,
        img_size=28,
    ):
        # random get local training and validating dataset
        (
            dataset_train,
            dataset_test,
            dict_users,
            test_users,
            skew_users,
        ) = util.dataset_loader(dataset_name, train_count, True, self.cluster_count)

        self.ldr_train = DataLoader(
            DatasetSplit(dataset_train, dict_users[self.port % self.cluster_count]),
            batch_size=10,
            shuffle=True,
        )

        self.ldr_test = DataLoader(
            DatasetSplit(dataset_test, test_users[self.port % self.cluster_count]),
            batch_size=10,
            shuffle=True,
        )

        # local train model
        self.local_model = util.model_loader(
            model_name, dataset_name, device, num_channels, num_classes, img_size
        )
        # this model to validate weight from leader
        self.validate_model = util.model_loader(
            model_name, dataset_name, device, num_channels, num_classes, img_size
        )

        self.nonce = 0

    def train_broadcast_w(self, local_ep=1, device="cpu", lr=0.003, local_bs=10):
        w_local, loss = train_new(
            self.local_model, self.ldr_train, local_ep, device, lr, local_bs
        )

        w_local_serialized = pickle.dumps(w_local)
        compressed_data = gzip.compress(w_local_serialized)
        b64_encoded = base64.b64encode(compressed_data)
        w_local_str = b64_encoded.decode("ascii")
        print(self.port, loss)

        # split model state_dict to transfer
        w_local_str_split = [
            w_local_str[:48000],
            w_local_str[48000:96000],
            w_local_str[96000:],
        ]
        # broadcast model state_dict
        for i in range(len(w_local_str_split)):
            data = {
                "type": "validate",
                "term": self.state.storage.term,
                "weight_serial": self.nonce,
                "weight_split_no": i,
                "weight_split": w_local_str_split[i],
            }
            self.broadcast(data)
        self.nonce += 1

    async def start(self):
        protocol = UDPProtocol(
            queue=self.requests, request_handler=self.request_handler, loop=self.loop
        )
        address = self.host, self.port
        self.transport, _ = await asyncio.Task(
            self.loop.create_datagram_endpoint(protocol, local_addr=address),
            loop=self.loop,
        )
        self.state.start()

    def stop(self):
        self.state.stop()
        self.transport.close()

    def update_cluster(self, address_list):
        self.cluster.update({address_list})

    @property
    def cluster_count(self):
        return len(self.cluster)

    def request_handler(self, data):
        self.state.request_handler(data)

    async def send(self, data, destination):
        """Sends data to destination Node
        Args:
            data — serializable object
            destination — <str> '127.0.0.1:8000' or <tuple> (127.0.0.1, 8000)
        """
        if isinstance(destination, str):
            host, port = destination.split(":")
            destination = host, int(port)

        await self.requests.put({"data": data, "destination": destination})

    def broadcast(self, data):
        """Sends data to all Nodes in cluster (cluster list does not contain self Node)"""
        for destination in self.cluster:
            asyncio.ensure_future(self.send(data, destination), loop=self.loop)
