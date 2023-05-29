from concurrent import futures
import threading
import logging
import config as cfg
from utils import decode, get_hash, save_chunk, divide_model, deliver_model, deliver_block, verify, select, get_chunk
from proto.node_pb2 import *
import torch
from obj import Model
import time, os, sys, gc
import grpc
from proto.node_pb2_grpc import NodeServiceStub
from proto.node_pb2 import * 


class Verification(threading.Thread):
    """
    This thread starts N-1 sub-threads to request votes.
    :param string sender: sender
    :param list addrs: the addresses of receivers
    :param string global_hash: the global model hash
    """
    def __init__(self, sender, addrs, global_hash, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.sender = sender
        self.addrs = addrs
        self.global_hash = global_hash
        self.result = False
        self.score_list = []

    def run(self): 
        try:
            self.approve = 0
            self.disapprove = 0
            with futures.ThreadPoolExecutor(max_workers=len(self.addrs)) as executor:
                res = []
                msg_sz = (cfg.hash_size+cfg.ec_n*(cfg.hash_size+8))*8/1000/1000
                recv_latency = cfg.net.sim_gossip_latency(cfg.ip_to_id[self.sender], msg_sz)
                for conn in self.addrs:
                    if self.sender == conn:
                        continue
                    res.append(executor.submit(verify, self.sender, conn, self.global_hash, recv_latency[cfg.ip_to_id[conn]]))
                for future in futures.as_completed(res):
                    response = future.result()
                    if response[0] == "Approve":
                        self.approve += 1
                        self.score_list.append(response[1])
                        if self.approve >= int(cfg.n_aggregators/2):
                            self.result = True
                            for r in res:
                                r.cancel()
                            executor.shutdown(wait=False)
                            logging.info("I win with {} approves and {} disapproves".format(self.approve, self.disapprove))
                            break
                    else:
                        self.disapprove += 1
                        if self.disapprove > int(cfg.n_aggregators/2):
                            for r in res:
                                r.cancel()
                            executor.shutdown(wait=False)
                            break

        finally:
            pass
        
    def get_result(self):
        return [self.result, self.score_list]

    def raise_exception(self):
        print("Verification Interuptted")


class BlockDelivery(threading.Thread):
    """
    This thread starts cfg.n_nodes-1 sub-threads to broadcast a block to all other nodes.
    :param string sender: the ip of the miner/sender of the block
    :param Block block
    """

    def __init__(self, sender, block, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.sender = sender
        self.block = block

    def run(self):
        counter = 0
        with futures.ThreadPoolExecutor(max_workers=cfg.n_nodes) as executor:
            res = []
            msg_sz = cfg.block_size*8/1000/1000
            recv_latency = cfg.net.sim_gossip_latency(cfg.ip_to_id[self.sender], msg_sz)
            print("Block broadcast delay {}s".format(max(recv_latency.values())))
            for conn in self.block.stake:
                future = executor.submit(deliver_block, conn, self.block, recv_latency[cfg.ip_to_id[conn]])
                res.append(future)
            for future in futures.as_completed(res):
                response = future.result()
                if response == "Success":
                    counter += 1
                if counter == cfg.n_nodes:
                    print("have sent to all nodes")


class ModelDelivery(threading.Thread):
    """
    This thread starts len(self.addrs) sub-threads to broadcast a model to all other nodes.
    :param list addrs: the addresses of receivers
    :param Model model
    """

    def __init__(self, sender, addrs, model, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.sender = sender
        self.addrs = addrs
        self.model = model

    def run(self):
        try:
            counter = 0
            with futures.ThreadPoolExecutor(max_workers=len(self.addrs)) as executor:
                res = []
                msg_sz = ( (len(self.model.chunks)+1)*(cfg.nodeid_size+cfg.hash_size)+len(self.model.scores)*(8+cfg.nodeid_size) )*8/1000/1000
                recv_latency = cfg.net.sim_gossip_latency(cfg.ip_to_id[self.sender], msg_sz)
                print("Model broadcast delay {}s".format(max(recv_latency.values())))
                for conn in self.addrs:
                    future = executor.submit(deliver_model, conn, self.model, recv_latency[cfg.ip_to_id[conn]])
                    res.append(future)
                for future in futures.as_completed(res):
                    response = future.result()
                    if response == "Success":
                        counter += 1
                    if counter == len(self.addrs):
                        executor.shutdown(wait=False)
    
        finally:
            pass

    def raise_exception(self):
        print("ModelDelivery exception")

class ModelRetreival(threading.Thread):
    """
    This thread starts cfg.ec_n sub-threads to retreive chunks to recover a model.
    :param Model model: the model to be recovered
    :param string fn: the file to save the recovered model
    """

    def __init__(self, tdf, ip, keepers, indexes, chunk_hashes, fn, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.tdf = tdf
        self.ip = ip
        self.keepers = keepers
        self.indexes = indexes
        self.chunk_hashes = chunk_hashes
        self.fn = fn
        self.scores = {}
        self.flag = False
    
    def run(self):
        try:
            index_chunk = []
            chunks = [None]*cfg.ec_n
            with futures.ThreadPoolExecutor(max_workers=cfg.ec_k) as executor:
                dsts = []
                counter = 0
                for idx, ck in enumerate(self.keepers):
                    dsts.append(cfg.ip_to_id[ck])
                msg_sz = cfg.chunk_size*8/1000/1000
                recv_latency = cfg.net.sim_get_or_save_chunk(cfg.ip_to_id[self.ip], len(dsts), dsts, msg_sz)
                print("Get chunk delay {}s".format(max(recv_latency.values())))
                counter = 0
                for idx in range(len(self.keepers)):
                    future = executor.submit(get_chunk, self.ip, self.keepers[idx], self.indexes[idx], self.chunk_hashes[idx], recv_latency[idx])
                    index_chunk.append(future)
                
                counter = 0
                for future in futures.as_completed(index_chunk):
                    res = future.result()
                    chunks[res[0]] = res[1]
                    self.scores[res[2]] = res[3]
                    counter += 1
                    if counter == cfg.ec_k:
                        print("get enough chunks")
            if not self.flag:
                my_model = decode(cfg.ec_k, cfg.ec_n, chunks)
                # assert get_hash(my_model) == self.model.model_hash
                with open(self.fn, 'wb') as f:
                    print("recover model {}".format(self.fn))
                    f.write(my_model)

        finally:
            pass

    def get_result(self):
        return self.scores

    def raise_exception(self):
        self.flag = True
        print("ModelRetreival exception end")


class ModelSave(threading.Thread):
    """
    This thread starts cfg.ec_n sub-threads to save a model.
    :param string fn: the model file
    """

    def __init__(self, ip, rd, sk, block_id, stake, fn, scores, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.ip = ip
        self.rd = rd
        self.sk = sk
        self.block_id = block_id
        self.stake = stake
        self.fn = fn
        self.scores = scores
        self.model = None

    def run(self):
        try:
            # divide local model into ec_n chunks
            model_hash, chunk_obj, chunks = divide_model(self.fn)
            keepers, _, _ = select(self.stake, cfg.ec_n, bytes(model_hash, encoding='utf-8'), self.sk)
            idx_map = {}
            for index, chunk in enumerate(chunks):
                chunk_obj[index].keeper = keepers[index]
                idx_map[keepers[index]] = index
    
            # start ec_n threads to save local model
            counter = 0
            with futures.ThreadPoolExecutor(max_workers=cfg.ec_n) as executor:            
                res = []
                dsts = []
                for index, conn in enumerate(keepers):
                    dsts.append(cfg.ip_to_id[chunk_obj[index].keeper])
                msg_sz = cfg.chunk_size*8/1000/1000
                recv_latency = cfg.net.sim_get_or_save_chunk(cfg.ip_to_id[self.ip], cfg.ec_n, dsts, msg_sz)
                print("Save chunk delay {}s".format(max(recv_latency.values())))

                for index, conn in enumerate(keepers):
                    logging.info("Save chunk {} in {}".format(chunk_obj[index].chunk_hash, chunk_obj[index].keeper))
                    future = executor.submit(save_chunk, self.ip, chunk_obj[index].keeper, model_hash, chunk_obj[index].chunk_hash, cfg.btl+self.block_id, chunks[index], recv_latency[index])
                    res.append(future)

                for future in futures.as_completed(res):
                    myresult = future.result()
                    # chunk_obj[idx_map[myresult[2]]].score = self.tdf-myresult[3]

                logging.info("Save model {} successfully".format(model_hash))
                self.model = Model(owner=self.ip, rd=self.rd, model_hash=model_hash, scores=self.scores, chunks=chunk_obj)
            
        finally:
            pass

    def get_result(self):
        return self.model

    def raise_exception(self):
        print("ModelSave exception end")
