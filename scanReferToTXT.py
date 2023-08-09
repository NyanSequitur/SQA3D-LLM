# load json files
import json

train = json.load(open('ScanRefer_filtered_train.json'))
test = json.load(open('ScanRefer_filtered_test.json'))
val = json.load(open('ScanRefer_filtered_val.json'))

sets = [train, test, val]

for set in sets:
    for caption in set:
        with open(f'GTCaptions/{caption["scene_id"]}.txt', 'a') as f:
            f.write(caption['description'] + '\n')