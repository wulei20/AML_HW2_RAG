import json
import re
import random
with open("preprocess/result.txt", 'r') as f:
    messages = []
    while True:
        prompt = f.readline()
        if not prompt:
            break
        result = f.readline()
        fixed_data = re.sub(r"(?<!\\)'", '"', prompt.replace('"', '\\"')).replace("\\'", "'")
        prompt = json.loads(fixed_data)
        for key in prompt[0]:
            if len(prompt[0][key]) > 511:
                prompt[0][key] = prompt[0][key][:511]
        prompt.append({"role": "assistant", "content": result[:511 if len(result) > 511 else len(result)]})
        messages.append({"messages": prompt})

train, dev = [], []

for item in messages:
    randval = random.random()
    if randval < 0.8:
        train.append(item)
    else:
        dev.append(item)

with open("data/train.jsonl", 'w') as f:
    for item in train:
        json.dump(item, f)
        f.write('\n')

with open("data/dev.jsonl", 'w') as f:
    for item in dev:
        json.dump(item, f)
        f.write('\n')