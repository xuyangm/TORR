from obj import Block, Model
from concurrent import futures
from utils import divide_model, save_chunk, deliver_block, select, read_from_file
import config as cfg
import secrets
import json
import torch
import time

from ML_models import init_model


def run():
    sk = secrets.token_bytes(nbytes=32)

    with open("genesis.json","r") as f:
        json_block = json.load(f)
    stakes = json_block["stake"]

    first_model = init_model(cfg.model_name)
    torch.save(first_model.state_dict(), "mymodel.pth")

    # select clients, aggregators and keepers
    _, beta_string, _ = select(stakes, cfg.n_clients+cfg.n_aggregators, bytes(json_block["id"]), sk)

    # divide the global model into chunks
    model_hash, chunk_obj, chunks = divide_model("mymodel.pth")
    print(len(model_hash), len(chunks[0]), len(read_from_file("mymodel.pth")))

    keepers, _, _ = select(stakes, len(chunks), model_hash.encode('utf-8'), sk)
    for idx, keeper in enumerate(keepers):
        chunk_obj[idx].keeper = keeper
    
    # start ec_n threads to save chunks to keepers
    counter = 0
    
    with futures.ThreadPoolExecutor(max_workers=cfg.ec_n) as executor:
        res = []
        for index, conn in enumerate(keepers):
            future = executor.submit(save_chunk, "", conn, model_hash, chunk_obj[index].chunk_hash, cfg.btl, chunks[index], 0)
            res.append(future)
        for future in futures.as_completed(res):
            response = future.result()
            if response[1] == "Success":
                counter += 1
            if counter == cfg.ec_n:
                executor.shutdown(wait=False)
                print("have saved all chunks")
                break
    
    # create a block and broadcast the block to other nodes
    counter = 0
    task_block = Block(timestamp=time.time(), time_diff=cfg.time_difficulty, rd=1, id=1, miner="model_owner", beta_string=beta_string, stake=stakes, models=[Model(owner="model_owner", model_hash=model_hash, scores={}, chunks=chunk_obj)], scores=[])
    with futures.ThreadPoolExecutor(max_workers=cfg.n_nodes) as executor:
        res = []
        for conn in stakes.keys():
            future = executor.submit(deliver_block, conn, task_block, 0)
            res.append(future)
        for future in futures.as_completed(res):
            response = future.result()
            if response == "Success":
                counter += 1
                print("Success", counter)
            if counter == cfg.n_nodes:
                executor.shutdown(wait=False)
                print("have sent block to every node")
                break


if __name__ == '__main__':
    run()