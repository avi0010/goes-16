import os
import json
import random

dir = './ttttt/hotspots'
files = os.listdir(dir)

files = random.sample(files, 11)
hotspots = json.load(open('./viirs/hotspots.json', 'r'))
for file in files:
    with open(os.path.join(dir, file), 'w') as f:
        json.dump(hotspots, f)