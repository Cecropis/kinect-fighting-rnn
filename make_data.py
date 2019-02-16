# -*- coding: utf-8 -*-
# 造数据
import os
import json
import random


origins = []
for i in os.walk('./Kinect_test/%s' % os.sys.argv[1]):
    for j in i[2]:
        with open(i[0] + '/' + j, 'r') as f:
            content = ''.join(f.readlines())
            ls = json.loads(content)
            origins.append({'dir': i[0], 'fname': j, 'ls': ls})

def task(item, j):
    for i in range(len(item['ls'])):
        for key in item['ls'][i]:
            if random.randint(0, 500) < 100:
                item['ls'][i][key] = [k * (1 - random.randint(-30, 25) / 1000) for k in item['ls'][i][key]]  # 小幅波动
            elif random.randint(0, 500) == 100:
                item['ls'][i][key] = [k * (1 - random.randint(-100, 125) / 1000) for k in item['ls'][i][key]]  # 大幅波动
        with open(item['dir'] + '/' + item['fname'].replace('.json', '_' + str(j)) + '.json', 'w') as f:
            f.write(json.dumps(item['ls'][random.randint(0, 10):]))  # 平移波形

for item in origins:
    for j in range(1, 2):
        task(item, j)