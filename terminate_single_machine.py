import os
import socket

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

IP = get_host_ip()
start = 50001
end = 50101
print('bash killcmd.sh %s %s' % (str(start), str(end)))
os.system('bash killcmd.sh %s %s' % (str(start), str(end)))
os.system('bash clean.sh')
os.system('screen -ls')