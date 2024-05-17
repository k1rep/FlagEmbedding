import json
input_data = []
with open('codesearchnet_shuf.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        input_data.append(json.loads(line))

output_data = [{"text": text} for text in input_data]
with open('output.jsonl', 'w', encoding='utf-8') as f:
    for item in output_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
