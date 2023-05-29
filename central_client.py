import grpc
import sys, os, time
from concurrent import futures
from proto.client_pb2 import *
from proto.client_pb2_grpc import add_ClientServiceServicer_to_server, ClientServiceServicer
import socket, json
from utils import submit_task, read_from_file, save_to_file

import config as cfg

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

class CentralClient(ClientServiceServicer):
    def __init__(self) -> None:
        super().__init__()
        self.ip = str(IP)+":"+str(sys.argv[1])
        with open("genesis.json","r") as f:
            json_block = json.load(f)
        stakes = json_block["stake"]
        addrs = list(stakes.keys())
        addrs.sort()
        for index, addr in enumerate(addrs):
            if addr == IP+":"+str(sys.argv[1]):
                print("my id is {}".format(index))
                self.id = index
                break
        self.rate = cfg.rates[int(self.id/20)]
        print("my rate is {} Mbps".format(self.rate))
        self.data_directory = str(sys.argv[1])+"record"
        os.system("mkdir %s" % self.data_directory)
        
    def LocalTrain(self, request, context):
        round = request.round
        global_model = request.model
        global_model_fn = self.data_directory+"/global_model-r"+str(round)+".pkl"
        save_to_file(global_model_fn, global_model)
        
        ML_server_ip = str(IP)+":9999"
        local_model_fn = self.data_directory+"/local_model-r"+str(round)+".pkl"
        res = submit_task(ML_server_ip, global_model_fn, local_model_fn, self.id)
        content = read_from_file(local_model_fn)
        os.system("rm %s" % global_model_fn)
        os.system("rm %s" % local_model_fn)
        
        return LocalModel(model=content)

    
def run():
    MAX_MESSAGE_LENGTH = 1024*1024*1024
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
               ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
               ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    add_ClientServiceServicer_to_server(CentralClient(),server)
    server.add_insecure_port(IP+":"+str(sys.argv[1]))
    server.start()
    print("Start central client at {}".format(IP+":"+str(sys.argv[1])))
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    run()