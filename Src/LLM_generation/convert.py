import os
import json
from collections import defaultdict

def process_filtered_file(input_path: str, output_path: str):
    query_id_to_items = defaultdict(list)

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            query_id = data.get("query_id")
            if query_id:
                query_id_to_items[query_id].append(data)

    # Filter query_ids with fewer than 30 entries, keep only one entry per query_id
    filtered_data = []
    for query_id, items in query_id_to_items.items():
        if len(items) < 30:
            filtered_data.append(items[0])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in filtered_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved {len(filtered_data)} entries to {output_path}")

if __name__ == "__main__":
    input_dir = "./Output/select"
    output_dir = "./Output/expand"
    langs = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh']

    for lang in langs:
        input_path = os.path.join(lang_dir, f"{lang}.jsonl")
        output_path = os.path.join(lang_dir, f"{lang}_expand.jsonl")

        if os.path.exists(input_path):
            process_filtered_file(input_path, output_path)
        else:
            print(f"File not found: {input_path}")
