from concurrent import futures
from utils import read_from_file, request_train, avg_model, save_to_file
import config as cfg
import json
import torch
from proto.client_pb2_grpc import ClientServiceStub
from proto.client_pb2 import * 
from ML_models import init_model
import random
import time, os


def run():
    id = 0

    with open("genesis.json","r") as f:
        json_block = json.load(f)
    stakes = json_block["stake"]

    first_model = init_model(cfg.model_name)
    torch.save(first_model.state_dict(), "mymodel.pth")
    content = read_from_file("mymodel.pth")
    nodes = list(stakes.keys())
    counter = 0
    
    start = time.time()
    current = 0
    with open("statistic/timestamp.txt", "a+") as f:
        f.write(str(current)+"\n")
    for i in range(1, cfg.n_rounds+1):
        print("############round {}###############".format(i))
        indx = 0
        local_model_fns = []
        clients = random.sample(nodes, cfg.n_clients)
        with futures.ThreadPoolExecutor(max_workers=len(clients)) as executor:
            res = []
            for conn in clients:
                future = executor.submit(request_train, conn, i, content)
                res.append(future)
            for future in futures.as_completed(res):
                response = future.result()
                local_model_fn = "local_model-r"+str(i)+"-"+str(indx)+".pkl"
                indx += 1
                save_to_file(local_model_fn, response)
                local_model_fns.append(local_model_fn)
        if id == 0:
            with open("statistic/chosen.txt", "a+") as f:
                n = [0, 0, 0, 0, 0]
                for v in clients:
                    n[int(cfg.ip_to_id[v] / 20)] += 1
                f.write(str(n[0])+" "+str(n[1])+" "+str(n[2])+" "+str(n[3])+" "+str(n[4])+"\n")
            
            for v in nodes:
                if v not in clients:
                    basic_store = 0
                else:
                    basic_store = cfg.model_size
                with open("statistic/node{}_storage.txt".format(cfg.ip_to_id[v]), "a+") as f:
                    f.write(str(basic_store/1000/1000)+"\n")
            with open("statistic/aggregator_storage.txt", "a+") as f:
                f.write(str(cfg.n_clients*cfg.model_size/1000/1000)+"\n")
        
        new_global_model = avg_model(local_model_fns)
        interval = time.time()-start
        current += interval
        start = time.time()
        global_model_fn = "global_model-r"+str(i+1)+".pkl"
        torch.save(new_global_model, global_model_fn)
        content = read_from_file(global_model_fn)
        with open("statistic/timestamp.txt", "a+") as f:
            f.write(str(current)+"\n")
        for fn in local_model_fns:
            os.system("rm %s" % fn)


if __name__ == '__main__':
    run()