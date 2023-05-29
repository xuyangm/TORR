import time
import sys
import os
import torch
import threading
from concurrent import futures
from ML_models import TwoNN
from obj import Block, Model
from data_manager import DatasetCreator
from utils import get_candidates, select, save_to_file, submit_task, get_model_hash, submit_agg
from encaped_process import BlockDelivery, ModelDelivery, ModelRetreival, ModelSave, Verification
import grpc
from proto.node_pb2 import *
from proto.node_pb2_grpc import add_NodeServiceServicer_to_server, NodeServiceServicer
import config as cfg
import logging
import vrf
import secrets
import random
import json
import socket
import numpy as np

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

IP = get_host_ip()

class NodeService(NodeServiceServicer):
    def __init__(self):
        self.previous = 0
        self.current = 0
        self.serious_lock = threading.Lock()
        self.chunk_lock = threading.Lock()
        self.thread_lock = threading.Lock()
        self.thread_pool = {}
        self.chunks = {}
        self.storage_capacity = 1024*1024*1024
        self.data_directory = str(sys.argv[1])+"record"
        os.system("mkdir %s" % self.data_directory)
        os.system("mkdir %s" % self.data_directory+"/chunks")
        os.system("mkdir %s" % self.data_directory+"/lks")
        os.system("mkdir %s" % self.data_directory+"/blocks")
        logging.basicConfig(filename=self.data_directory+"/"+str(sys.argv[1])+'.log', filemode='a', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        self.ip = str(IP)+":"+str(sys.argv[1])
        
        with open("genesis.json","r") as f:
            json_block = json.load(f)
        stakes = json_block["stake"]
        self.id = cfg.ip_to_id[self.ip]
        self.rate = cfg.rates[int(self.id/20)]
        self.blockchain = [Block()]
        self.models = {}
        self.state = "IDLE"
        self.role = None
        self.global_model_hash = None
        self.sk = secrets.token_bytes(nbytes=32)
        self.pk = vrf.get_public_key(self.sk)
        self.clients = []
        self.aggregators = []
        self.scores = {}


    def Verify(self, request, context):
        """
        The leader determined by Bully protocol may request verification from other aggregators.
        Should provide verification and scores to keepers.
        """ 
        response = VerifyResponse()  
        if self.global_model_hash is None:
            res = "Global model is not ready"
        elif self.state != "VOTE":
            res = "It is not the time for voting"
        elif self.global_model_hash != request.hash:
            res = "Receive inconsistent model hash"
        else:
            res = "Approve"

        response.result = res
        response.scores.update(self.scores)

        if res != "Approve":
            logging.info("Disapprove: {}".format(res)) 
        
        return response


    def DeliverBlock(self, request, context):
        logging.info("######## Receive block{} from {} ########".format(request.id, request.miner))
        bk = Block()
        bk.deserialize(request)
        # add the block to local ledger
        self.serious_lock.acquire()
        if self.blockchain[len(self.blockchain)-1].id+1 == bk.id and bk.rd <= cfg.n_rounds:
            self.blockchain.append(bk)
            with open(self.data_directory+"/blocks/"+"block"+str(bk.id), 'w') as f:
                json.dump(bk.to_json(), f, indent=4)

            if self.previous == 0:
                self.current = 0
            else:
                self.current += time.time()-self.previous
            self.previous = time.time()
            if self.id == 0:
                with open("statistic/timestamp.txt", "a+") as f:
                    f.write(str(self.current)+"\n")
            else:
                with open(self.data_directory+"/timestamp.txt", "a+") as f:
                    f.write(str(self.current)+"\n")

            # reset all
            for t in self.thread_pool:
                if t.is_alive():
                    t.raise_exception()
            self.thread_pool = []
            self.set_state("NEW ROUND")
            self.clean_outdated_chunks(bk.rd)
            self.stake = bk.stake
            self.global_model_hash = None
            chosen = get_candidates(self.stake, bk.beta_string, cfg.n_clients+cfg.n_aggregators)
            # tmp_stake = {}
            # for c in chosen:
            #     tmp_stake[c] = self.stake[c]
            # sorted_keys = sorted(tmp_stake, key=tmp_stake.get, reverse=True)
            self.clients = chosen[:cfg.n_clients]
            self.aggregators = chosen[cfg.n_clients:]
            logging.info("Aggregators: {}".format(self.aggregators))
            logging.info("Clients: {}".format(self.clients))

            if self.ip in self.clients:
                self.role = "client"
            elif self.ip in self.aggregators:
                self.role = "aggregator"
            else:
                self.role = "nobody"
            
            # record data
            if self.id == 0:
                with open("statistic/chosen.txt", "a+") as f:
                    n = [0, 0, 0, 0, 0]
                    for v in self.aggregators:
                        n[int(cfg.ip_to_id[v] / 20)] += 1
                    for v in self.clients:
                        n[int(cfg.ip_to_id[v] / 20)] += 1
                    f.write(str(n[0])+" "+str(n[1])+" "+str(n[2])+" "+str(n[3])+" "+str(n[4])+"\n")
                with open("statistic/stake.txt", "a+") as f:
                    s = [0, 0, 0, 0, 0]
                    for v in self.stake:
                        s[int(cfg.ip_to_id[v] / 20)] += self.stake[v]
                    for i in range(len(s)):
                        s[i] = s[i]/20
                    f.write(str(s[0])+" "+str(s[1])+" "+str(s[2])+" "+str(s[3])+" "+str(s[4])+"\n")
            with open("statistic/node{}_storage.txt".format(self.id), "a+") as f:
                basic_store = (len(self.chunks)*cfg.chunk_size+bk.rd*cfg.block_size)
                if self.role == "nobody":
                    f.write(str(basic_store/1000/1000)+"\n")
                elif self.role == "client":
                    f.write(str((basic_store+cfg.model_size)/1000/1000)+"\n")
                else:
                    f.write(str((basic_store+2*cfg.model_size)/1000/1000)+"\n")
            
            print("I'm", self.role)
            logging.info("I'm {}".format(self.role))
            self.serious_lock.release()

            if self.role == "client":
                # recover the global model
                global_model_fn = self.data_directory+"/"+bk.models[0].model_hash+".pkl"
                keepers = []
                indexes = []
                chunk_hashes = []
                for ck in bk.models[0].chunks:
                    keepers.append(ck.keeper)
                    indexes.append(ck.index)
                    chunk_hashes.append(ck.chunk_hash)
                    if len(keepers) == cfg.ec_k:
                        break
                thread = ModelRetreival(bk.time_diff, self.ip, keepers, indexes, chunk_hashes, global_model_fn)
                thread.start()
                thread.join()
                scores = thread.get_result()
                
                self.serious_lock.acquire()
                ML_server_ip = str(IP)+":9999"
                local_model_fn = self.data_directory+"/"+"local_model-r"+str(bk.rd)+".pkl"
                res = submit_task(ML_server_ip, global_model_fn, local_model_fn, self.id)
                assert res == "Success"
                
                # store the local model
                thread = ModelSave(self.ip, bk.rd, self.sk, self.blockchain[len(self.blockchain)-1].id, self.stake, local_model_fn, scores)
                thread.start()
                thread.join()
                local_model_object = thread.get_result()
                os.system("rm %s" % local_model_fn)
                self.serious_lock.release()
                
                # send the local model to aggregators
                thread = ModelDelivery(self.ip, self.aggregators, local_model_object)
                thread.start()

            elif self.role == "aggregator":
                # check whether we have enough local models
                if bk.rd in self.models and len(self.models[bk.rd]) == cfg.n_clients:
                    self.serious_lock.acquire()
                    self.aggregate(context)
                    self.serious_lock.release()
            

        elif bk.rd > cfg.n_rounds:
            print("Task finished!")
            self.serious_lock.release()

        else:
            self.serious_lock.release()
        
        return BlockResponse(result="Success")

    def DeliverModel(self, request, context):
        logging.info("######## Receive Model {} from {} ########".format(request.model_hash, request.owner))
        m = Model()
        m.deserialize(request)
        
        self.serious_lock.acquire()
        if m.rd in self.models:
            self.models[m.rd].append(m)
        else:
            # del self.models
            # n = gc.collect()
            # print("Collect {} garbages".format(n))
            self.models = {}
            self.models[m.rd] = [m]

        if self.blockchain[len(self.blockchain)-1].rd == m.rd and len(self.models[m.rd]) == cfg.n_clients:
            # if we have enough local models
            self.aggregate(context)
        self.serious_lock.release()
                
        return ModelResponse(result="Success")

    # PASS
    def SaveChunk(self, request, context):
        model_hash = request.model_hash
        chunk_hash = request.chunk_hash
        btl = request.btl
        content = request.content

        self.chunk_lock.acquire()
        if chunk_hash in self.chunks:
            fn = self.data_directory+"/chunks/"+chunk_hash+".dat"
            self.chunks[chunk_hash] = [fn, btl]
            result = "Success"
        else:
            chunk_size = len(content)
            if self.storage_capacity < chunk_size:
                result = "Out of storage"
            else:
                fn = self.data_directory+"/chunks/"+chunk_hash+".dat"
                save_to_file(fn, content)
                self.chunks[chunk_hash] = [fn, btl]
                self.storage_capacity -= chunk_size
                result = "Success"
        if result != "Success":
            print("Save chunk error: {}".format(result))
        self.chunk_lock.release()
        return SaveChunkResponse(model_hash=model_hash, result=result)

    # PASS
    def GetChunk(self, request, context):
        chunk_hash = request.hash
        self.chunk_lock.acquire()
        fn = self.chunks[chunk_hash][0]
        with open(fn, 'rb') as f:
            content = f.read()
        self.chunk_lock.release()
        return GetChunkResponse(content=content)
    
    def aggregate(self, context):
        logging.info("{} models have been collected.".format(cfg.n_clients))
        self.set_state("AGGREGATION")
        bk = self.blockchain[len(self.blockchain)-1]
        cur_rd = bk.rd
        # recover local models
        tpool = []
        local_model_fns = []
        self.serious_lock.release()
        
        for idx in range(len(self.models[bk.rd])):
            fn = self.data_directory+"/lks/"+str(self.models[bk.rd][idx].model_hash)+".pkl"
            local_model_fns.append(fn)
            keepers = []
            indexes = []
            chunk_hashes = []
            for ck in self.models[bk.rd][idx].chunks:
                keepers.append(ck.keeper)
                indexes.append(ck.index)
                chunk_hashes.append(ck.chunk_hash)
                if len(keepers) == cfg.ec_k:
                    break
            thread = ModelRetreival(bk.time_diff, self.ip, keepers, indexes, chunk_hashes, fn)
            thread.start()
            tpool.append(thread)
            self.thread_pool.append(thread)
        
        scores_list = {}
        for idx in range(len(tpool)):
            tpool[idx].join()
            tmp_scores = tpool[idx].get_result()
            
            for k in tmp_scores:
                if k not in scores_list:
                    scores_list[k] = [tmp_scores[k]]
                else:
                    scores_list[k].append(tmp_scores[k]) 
        
        self.serious_lock.acquire()
        if self.blockchain[len(self.blockchain)-1].rd > cur_rd:
            print("Receive another block during aggregation")
            self.serious_lock.release()
            context.abort(grpc.StatusCode.ABORTED, "Receive another block during aggregation")
        
        self.scores = {}
        for k in scores_list:
            self.scores[k] = np.mean(scores_list[k])
        
        global_model_fn = self.data_directory+"/"+"global_model-r"+str(bk.rd+1)+".pkl"
        ML_server_ip = str(IP)+":9999"
        res = submit_agg(ML_server_ip, local_model_fns, global_model_fn)
        sorted_stake = {}
        for n in self.aggregators:
            sorted_stake[n] = self.stake[n]
        sorted_tuple = sorted(sorted_stake.items(), key=lambda x: (x[1], -cfg.ip_to_id[x[0]]), reverse=True)
        print("Who's the big brother? {}!".format(sorted_tuple[0][0]))
        
        for fn in local_model_fns:
            os.system("rm %s" % fn)

        self.global_model_hash = get_model_hash(global_model_fn)
        self.set_state("VOTE")
        if sorted_tuple[0][0] == self.ip:
            print("I'm the leader!")
            new_block = Block(timestamp=time.time(), time_diff=bk.time_diff, rd=bk.rd+1, id=bk.id+1, miner=self.ip)
            if bk.rd % cfg.delta == 0:
                time_costs = []
                num = cfg.delta
                step = 1
                while num > 0 and len(self.blockchain)-step > 0:
                    if self.blockchain[len(self.blockchain)-step].id == 1:
                        num -= 1
                        step += 1
                        continue
                                
                    for s in self.blockchain[len(self.blockchain)-step].scores:
                        time_costs.append(s)
                    num -= 1
                    step += 1
                
                if len(time_costs) > 0:
                    new_time_diff = np.median(time_costs)
                    logging.info("Tweak the time difficulty to {}".format(new_time_diff))
                    new_block.time_diff = new_time_diff
                        
            count = 0
            verified = False
            final_scores = {}
            while not verified:
                count += 1
                thread = Verification(self.ip, self.aggregators, self.global_model_hash)
                thread.start()
                thread.join()
                result = thread.get_result()
                verified = result[0]
                if not verified:
                    if count < 10:
                        time.sleep(count)
                    else:
                        time.sleep(10)
                else:
                    score_list = result[1]
                    for m in self.models[bk.rd]:
                        score_list.append(m.scores)
                    score_list.append(self.scores)
                    
                    new_block.scores = []
                    
                    for score_evaluation in score_list:
                        for k in score_evaluation:
                            new_block.scores.append(score_evaluation[k])
                            if k not in final_scores:
                                final_scores[k] = [score_evaluation[k]]
                            else:
                                final_scores[k].append(score_evaluation[k])
                    # if self.blockchain[len(self.blockchain)-1].rd > 1:
                    self.update_stake(final_scores, self.blockchain[len(self.blockchain)-1].time_diff)
                print("use {} terms".format(count))
            
            thread = ModelSave(self.ip, bk.rd+1, self.sk, bk.id+1, self.stake, global_model_fn, {})
            thread.start()
            thread.join()
            global_model_object = thread.get_result()
            _, beta_string, _ = select(self.stake, cfg.n_clients+cfg.n_aggregators, bytes(new_block.id), self.sk)
            new_block.beta_string = beta_string
            new_block.stake = self.stake
            new_block.models = [global_model_object]
            
            logging.info("I am the leader!")
            os.system("cp %s ./" % global_model_fn)
            thread = BlockDelivery(self.ip, new_block)
            thread.start()

    def clean_outdated_chunks(self, rd):
        self.chunk_lock.acquire()
        for saved_chunk in list(self.chunks):
            fn = self.chunks[saved_chunk][0]
            deadline = self.chunks[saved_chunk][1]
            if rd > deadline:
                os.system('rm %s' % fn)
                self.chunks.pop(saved_chunk)
        self.chunk_lock.release()

    def set_state(self, state):
        logging.info("{}->{}".format(self.state, state))
        self.state = state

    def update_stake(self, final_scores, time_diff):
        print("update stake #######################:")
        for keeper in final_scores:
            evaluation = np.median(final_scores[keeper])
            print(keeper, ":", evaluation)
            # self.stake[keeper] += time_diff-evaluation
            self.stake[keeper] += 1.0/evaluation
            # if self.stake[keeper] <= 0:
            #     self.stake[keeper] = 0.4

def run():
    MAX_MESSAGE_LENGTH = 1024*1024*1024
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1000), options=[
               ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
               ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    add_NodeServiceServicer_to_server(NodeService(),server)
    server.add_insecure_port(IP+":"+str(sys.argv[1]))
    server.start()
    print("Start service at {}".format(IP+":"+str(sys.argv[1])))
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
    # time.sleep(60)

if __name__ == '__main__':
    run()
