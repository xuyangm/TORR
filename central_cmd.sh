screen -L -S 9999 -t 9999 -dm python3 ML_server.py
for i in `seq $1 $2`
do
    screen -L -S $i -t $i -dm python3 central_client.py $i
done
