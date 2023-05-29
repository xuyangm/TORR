from obj import Block
import json


stakes = {}
for line in open('bootstrap.txt', 'r'):
    rs = line.rstrip('\n') 
    stakes[rs] = 1

genesis = Block(stake=stakes, models=[])
json_block = genesis.to_json()

with open("genesis.json", 'w') as f:
    json.dump(json_block, f, indent=4)