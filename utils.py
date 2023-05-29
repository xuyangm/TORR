import zfec
import hashlib
import copy
import torch
import config as cfg
from obj import Chunk
import grpc
from proto.node_pb2_grpc import NodeServiceStub
from proto.node_pb2 import * 
from proto.ML_pb2_grpc import MLServiceStub
from proto.ML_pb2 import *
from proto.client_pb2_grpc import ClientServiceStub
from proto.client_pb2 import * 
import vrf
from timeit import default_timer as timer
import time

from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import gc, sys

#####################
#    zfec encode    #
#####################
def encode(k, n, m):
    """Erasure encodes string ``m`` into ``n`` blocks, such that any ``k``
    can reconstruct.
    :param int k: k
    :param int n: number of blocks to encode string ``m`` into.
    :param bytes m: bytestring to encode.
    :return list: Erasure codes resulting from encoding ``m`` into
        ``n`` blocks using ``zfec`` lib.
    """
    try:
        m = m.encode()
    except AttributeError:
        pass
    encoder = zfec.Encoder(k, n)
    assert k <= 256  # TODO: Record this assumption!
    # pad m to a multiple of K bytes
    padlen = k - (len(m) % k)
    m += padlen * chr(k - padlen).encode()
    step = len(m) // k
    blocks = [m[i * step : (i + 1) * step] for i in range(k)]
    stripes = encoder.encode(blocks)
    return stripes


def decode(k, n, stripes):
    """Decodes an erasure-encoded string from a subset of stripes
    :param list stripes: a container of :math:`n` elements,
        each of which is either a string or ``None``
        at least :math:`k` elements are strings
        all string elements are the same length
    """
    assert len(stripes) == n
    blocks = []
    blocknums = []
    for i, block in enumerate(stripes):
        if block is None:
            continue
        blocks.append(block)
        blocknums.append(i)
        if len(blocks) == k:
            break
    else:
        raise ValueError("Too few to recover")
    decoder = zfec.Decoder(k, n)
    rec = decoder.decode(blocks, blocknums)
    m = b"".join(rec)
    padlen = k - m[-1]
    m = m[:-padlen]
    return m


def get_hash(x):
    assert isinstance(x, (str, bytes))
    try:
        x = x.encode()
    except AttributeError:
        pass
    return hashlib.sha256(x).hexdigest() #hashlib.sha256(x).digest()

def get_model_hash(fn):
    with open(fn, "rb") as f:
        content = f.read()
    return get_hash(content)


def divide_model(fn):
    """
    Divide a model into chunks
    :param string fn: the file of the model

    Return:
    string model_hash: the hash of this model
    list chunk_obj: a list of chunk object
    list chunks: a list of chunks
    """
    f = open(fn, "rb")
    content = f.read()
    f.close()
    # set global model hash
    model_hash = get_hash(content)
    # divide model into n chunks
    chunks = encode(cfg.ec_k, cfg.ec_n, content)
    chunk_obj = []
    for index, c in enumerate(chunks):
        chunk_obj.append(Chunk(chunk_hash=get_hash(c), index=index))

    return model_hash, chunk_obj, chunks

def grpc_server_on(channel) -> bool:
    try:
        grpc.channel_ready_future(channel).result(timeout=30)
        return True
    except grpc.FutureTimeoutError:
        return False

def save_chunk(src, to, model_hash, chunk_hash, btl, content, delay):
    start = timer()
    time.sleep(delay)
    with grpc.insecure_channel(to) as channel:
        if not grpc_server_on(channel):
            print("Save chunk failed, server {} not started".format(to))
            return ["", "Failed"]
        stub = NodeServiceStub(channel)
        req = SaveChunkRequest()
        req.model_hash = model_hash
        req.chunk_hash = chunk_hash
        req.btl = btl
        req.content = content
        rmodel_hash = None
        rresult = None
        try:
            response = stub.SaveChunk(req)
            rmodel_hash = response.model_hash
            rresult = response.result
        except grpc.RpcError as e:
            e.details()
            print(e)
    
    end = timer()
    return [rmodel_hash, rresult, to, end-start]

def get_chunk(src, to, index, chunk_hash, delay):
    start = timer()
    time.sleep(delay)
    with grpc.insecure_channel(to) as channel:
        if not grpc_server_on(channel):
            print("Get chunk failed, server {} not started".format(to))
            return [0, b'']
        stub = NodeServiceStub(channel)
        req = GetChunkRequest()
        req.hash = chunk_hash
        rcontent = b''
        try:
            response = stub.GetChunk(req)
            rcontent = response.content
        except grpc.RpcError as e:
            e.details()
            print(e)
    
    end = timer()
    return [index, rcontent, to, end-start]

def deliver_block(ip, block, delay):
    time.sleep(delay)
    with grpc.insecure_channel(ip) as channel:
        if not grpc_server_on(channel):
            print("Deliver block failed, server {} not started".format(ip))
            return "Failed"
        stub = NodeServiceStub(channel)
        req = grpcBlock()
        block.serialize(req)
        result = ""
        try:
            response = stub.DeliverBlock(req)
            result = str(response.result)
        except grpc.RpcError as e:
            pass
    return result

def deliver_model(ip, model, delay):
    time.sleep(delay)
    with grpc.insecure_channel(ip) as channel:
        if not grpc_server_on(channel):
            print("Model deliver failed, server {} not started".format(ip))
            return "Failed"
        stub = NodeServiceStub(channel)
        req = grpcModel()
        model.serialize(req)
        rresult = ""
        try:
            response = stub.DeliverModel(req)
            rresult = str(response.result)
        except grpc.RpcError as e:
            pass
    return rresult

def verify(src, to, h, delay):
    time.sleep(delay)
    with grpc.insecure_channel(to) as channel:
        if not grpc_server_on(channel):
            print("Model deliver failed, server {} not started".format(to))
            return "Failed"
        stub = NodeServiceStub(channel)
        req = VerifyRequest(hash=h)
        rresult = ""
        scores = {}
        try:
            response = stub.Verify(req)
            rresult = str(response.result)
            scores = response.scores
        except grpc.RpcError as e:
            e.details()
            print(e)
    
    return [rresult, scores]

def submit_task(ip, fn, to_save, node_id):
    with grpc.insecure_channel(ip) as channel:
        if not grpc_server_on(channel):
            print("ML server not started")
            return "Failed"
        stub = MLServiceStub(channel)
        req = TaskRequest(fn=fn, to_save=to_save, node_id=node_id)
        rresult = ""
        try:
            response = stub.SubmitMLTask(req)
            rresult = str(response.result)
        except grpc.RpcError as e:
            e.details()
            print(e)
    return rresult

def submit_test(ip, fn):
    with grpc.insecure_channel(ip) as channel:
        if not grpc_server_on(channel):
            print("ML server not started")
            return "Failed"
        stub = MLServiceStub(channel)
        req = TestRequest(fn=fn)
        rresult = [0, 0]
        try:
            response = stub.SubmitTest(req)
            rresult[0] = response.accuracy
            rresult[1] = response.loss
        except grpc.RpcError as e:
            e.details()
            print(e)
    return rresult

def submit_agg(ip, fns, fn):
    with grpc.insecure_channel(ip) as channel:
        if not grpc_server_on(channel):
            print("ML server not started")
            return "Failed"
        stub = MLServiceStub(channel)
        req = AggregateRequest()
        req.fns.extend(fns)
        req.fn = fn
        rresult = ""
        try:
            response = stub.SubmitAggTask(req)
            rresult = response.fn
        except grpc.RpcError as e:
            e.details()
            print(e)
    return rresult

def request_train(ip, round, content):
    # bandwidth = cfg.rates_map[ip]
    # msg_sz = 2*len(content)*8/1000/1000
    # delay = msg_sz/bandwidth
    # print("msg size {} Mb, bandwidth {} Mbps, delay {} s".format(msg_sz, bandwidth, delay))
    # time.sleep(delay)
    
    with grpc.insecure_channel(ip, options=[
        ('grpc.max_send_message_length', 1024*1024*1024),
        ('grpc.max_receive_message_length', 1024*1024*1024),
    ]) as channel:
        if not grpc_server_on(channel):
            print("central client not started")
            return "Failed"
        stub = ClientServiceStub(channel)
        req = GlobalModel(round=round, model=content)
        rresult = b''
        try:
            response = stub.LocalTrain(req)
            rresult = response.model
        except grpc.RpcError as e:
            e.details()
            print(e)
    return rresult

def select(nodes, num, public_str, sk):
    """
    Select `num` candidates from `vnodes` through vrf
    :param dict nodes: nodes and their stakes. To be simple, we use ip:port to identify a node
    :param int num: the number of candidates to be selected
    :param bytes public_str: a public string used for vrf. To be simple, we use the id of previous block
    :param string sk: secret key

    Return:
    list chosen: selected nodes
    bytes beta_string: shared with other nodes. Should select the same hashes if using this string
    bytes pi_string: used for proof
    """
    hash_ip = {}
    
    available_nodes = 0
    for k, v in nodes.items():
        if round(v) > 0:
            available_nodes += 1
    coefficient = 1
    print("available_nodes: {}".format(available_nodes))
    if available_nodes < num:
        print("ERROR! NOT ENOUGH!")
        coefficient = 2
    
    for k, v in nodes.items():
        for i in range(round(v*coefficient)):
            node = k+str(i)
            hash_ip[get_hash(node)] = k

    vnodes = list(hash_ip.keys())
    vnodes.sort()
    chosen = {}

    p_status, pi_string = vrf.ecvrf_prove(sk, public_str)
    b_status, beta_string = vrf.ecvrf_proof_to_hash(pi_string)
    if p_status != "VALID" or b_status != "VALID":
        print("vrf error")
        exit(-1)

    cs = get_hash(beta_string) 

    while len(chosen) != num:
        pos = _find_pos(vnodes, cs)
        if pos == len(vnodes):
            pos = 0
        if hash_ip[vnodes[pos]] not in chosen:
            chosen[hash_ip[vnodes[pos]]] = True
        cs = get_hash(cs)
    
    return list(chosen.keys()), beta_string, pi_string

def get_candidates(nodes, beta_string, num):
    hash_ip = {}
    
    available_nodes = 0
    for k, v in nodes.items():
        if round(v) > 0:
            available_nodes += 1
    coefficient = 1
    if available_nodes < num:
        print("ERROR! NOT ENOUGH!")
        coefficient = 2
    
    for k, v in nodes.items():
        for i in range(round(v*coefficient)):
            node = k+str(i)
            hash_ip[get_hash(node)] = k

    vnodes = list(hash_ip.keys())
    vnodes.sort()
    chosen = {}

    cs = get_hash(beta_string)

    while len(chosen) != num:
        pos = _find_pos(vnodes, cs)
        if pos == len(vnodes):
            pos = 0
        if hash_ip[vnodes[pos]] not in chosen:
            chosen[hash_ip[vnodes[pos]]] = True
        cs = get_hash(cs)
    
    return list(chosen.keys())


def _find_pos(sorted_hash_candidates, h):
    left = 0 
    right = len(sorted_hash_candidates) - 1
    while left <= right:
        middle = int((left + right) / 2)
        if sorted_hash_candidates[middle] < h:
            left = middle + 1
        elif sorted_hash_candidates[middle] > h:
            right = middle - 1
        else:
            return middle
    return right + 1


def save_to_file(fn, content):
    with open(fn, "wb") as f:
        f.write(content)


def read_from_file(fn):
    with open(fn, "rb") as f:
        content = f.read()
    return content

def avg_model(local_model_fns):
    local_model_fns.sort()
    local_models = []
    for fn in local_model_fns:
        local_models.append(torch.load(fn))
    model_state = copy.deepcopy(local_models[0])
    for idx, local_model in enumerate(local_models):
        if idx == 0:
            continue
        for key in local_model.keys():
            model_state[key] = torch.add(model_state[key], local_model[key])

    for key in model_state.keys():
        model_state[key] = torch.div(model_state[key], len(local_models))
    
    del local_models
    return model_state


def get_dataloader():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    batch_size = cfg.batch_size
    trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform_train)
    sampler = RandomSampler(data_source=trainset, num_samples=500, replacement=True)
    trainloader = DataLoader(trainset, sampler=sampler, batch_size=batch_size, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, testloader