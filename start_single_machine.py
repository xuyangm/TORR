import os
import socket
import config as cfg

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

IP = get_host_ip()
rate = int(cfg.rates_map[str(IP)+":50001"]*1000)
print(rate)
start = 50001
end = 50020
print('bash cmd.sh %s %s %s' % (str(start), str(end), str(rate)))
os.system('bash cmd.sh %s %s %s' % (str(start), str(end), str(rate)))
