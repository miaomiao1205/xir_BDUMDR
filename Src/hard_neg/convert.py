import json
import os

# Step 1: Load corpus file
def load_corpus_jsonl(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line.strip())
            data[doc['docid']] = doc
    return data

# Step 2: Load candidate negative IDs
def build_negative_passages(negative_file_path, corpus_data):
    result = {}
    with open(negative_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            negative = json.loads(line.strip())
            query_id = negative['query_id']
            docid_id = negative['docid_id']

            if docid_id in corpus_data:
                doc = corpus_data[docid_id]
                passage = {
                    "docid": doc.get("docid", ""),
                    "title": doc.get("title", ""),
                    "text": doc.get("text", "")
                }

                if query_id not in result:
                    result[query_id] = {
                        "query_id": query_id,
                        "negative_passages": []
                    }
                result[query_id]["negative_passages"].append(passage)
    return result

# Step 3: Load MIRACL training data
def load_miracl_jsonl(file_path):
    miracl_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            miracl_data[data['query_id']] = data
    return miracl_data

# Step 4: Replace negative_passages in MIRACL training set
def replace_negative_passages(miracl_data, hard_negatives_data):
    for query_id, miracl_record in miracl_data.items():
        if query_id in hard_negatives_data:
            miracl_record['negative_passages'] = hard_negatives_data[query_id]['negative_passages']
    return miracl_data

# Step 5: Save to final output file
def save_to_new_file(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for record in data.values():
            file.write(json.dumps(record, ensure_ascii=False) + '\n')

def main():
    langs = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh']
    param = "xxx" 
    for lang in langs:
        try:
            corpus_file = f'./Data/miracl/corpus/{lang}.jsonl'
            hard_ids_file = f'./Output/rank/{lang}/{lang}_{param}.txt'
            miracl_query_file = f'./Data/miracl/train/{lang}.jsonl'
            final_output_file = f'./Output/candidate/{lang}.jsonl'

            if not os.path.exists(corpus_file):
                print(f'File {corpus_file} does not exist. Skipping {lang}.')
                continue
            if not os.path.exists(hard_ids_file):
                print(f'File {hard_ids_file} does not exist. Skipping {lang}.')
                continue
            if not os.path.exists(miracl_query_file):
                print(f'File {miracl_query_file} does not exist. Skipping {lang}.')
                continue

            corpus_data = load_corpus_jsonl(corpus_file)

            hard_negatives_data = build_negative_passages(hard_ids_file, corpus_data)

            miracl_data = load_miracl_jsonl(miracl_query_file)

            updated_miracl_data = replace_negative_passages(miracl_data, hard_negatives_data)

            save_to_new_file(updated_miracl_data, final_output_file)

            print(f'Finished processing language: {lang}')

        except Exception as e:
            print(f'Error while processing {lang}: {e}')
            continue

if __name__ == "__main__":
    main()