with open("bootstrap.txt", 'w') as f:
    for i in range(50001, 50101):
        f.write("172.17.0.10:"+str(i)+"\n")
