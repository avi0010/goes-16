import os
import json
from itertools import combinations
from datetime import datetime

json_files = [os.path.join('geojson', file) for file in os.listdir('geojson')]

def getOverlap(a, b):
    x = max(a[0], b[0])
    y = min(a[1], b[1])

    return min(a[1], b[1]) - max(a[0], b[0])

file_combinations = list(combinations(json_files, 2))
for file_pair in file_combinations:
    file_1, file_2 = file_pair

    first_json = json.load(open(file_1, 'r'))['features'][0]['properties']
    second_json = json.load(open(file_2, 'r'))['features'][0]['properties']

    first_interval = datetime.strptime(first_json['start_date'], '%Y-%m-%dT%H:%M:%SZ'), datetime.strptime(first_json['end_date'], '%Y-%m-%dT%H:%M:%SZ')
    second_interval = datetime.strptime(second_json['start_date'], '%Y-%m-%dT%H:%M:%SZ'), datetime.strptime(second_json['end_date'], '%Y-%m-%dT%H:%M:%SZ')

    print(f'{os.path.basename(file_1)} : {os.path.basename(file_2)}\t', getOverlap(first_interval, second_interval))