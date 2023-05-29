ec_k = 15
ec_n = 33
btl = 10
delta = 1

n_nodes = 100
n_clients = 10
n_aggregators = 26
n_rounds = 101
nodeid_size = 64
hash_size = 64
chunk_size = 16726 #373741 #686718 #16726
model_size = 250890 #5606109 #10300761 #250890
block_size = 12036

time_difficulty = 3

batch_size = 32
model_name = "LeNet"#"ShuffleNetV2" #"MobileNetV3" #"LeNet"

rates = [50, 25, 10, 1, 0.2] # Wifi, 4G+, 4G, LTE-M, NB-IoT # Mbps

ip_to_id = {}
rates_map = {}
count = 0
for line in open('bootstrap.txt', 'r'):
    rs = line.rstrip('\n')
    ip_to_id[rs] = count
    rates_map[rs] = rates[int(count/20)]
    count += 1

from network import Network
net = Network()
