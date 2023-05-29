screen -S 9999 -X quit
for i in `seq $1 $2`
do
    screen -S $i -X quit
done
