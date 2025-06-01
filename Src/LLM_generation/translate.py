import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import translate_v2 as translate

INPUT_FILE = "./Data/alpaca/alpaca.jsonl"
OUTPUT_FILE = "./Output/alpaca/alpaca_all.jsonl"
LANGS = ['ar', 'bn', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh']
BATCH_SIZE = 20

# Set your path to Google Cloud credentials JSON file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
client = translate.Client()

def translate_batch(texts: list, target_lang: str) -> list:
    try:
        results = client.translate(
            texts,
            target_language=target_lang,
            format_='text',
            model='nmt'
        )
        return [result['translatedText'] for result in results]
    except Exception as e:
        print(f"[{target_lang}] Batch translation failed: {e}")
        return texts  # Return original texts in case of failure

def process_language(entries: list, lang: str) -> list:
    translated_entries = []

    instructions = [e.get("instruction", "") for e in entries]
    inputs = [e.get("input", "") for e in entries]
    outputs = [e.get("output", "") for e in entries]

    translated_instructions = translate_batch(instructions, lang)
    translated_inputs = translate_batch(inputs, lang)
    translated_outputs = translate_batch(outputs, lang)

    for orig, ti, tin, tout in zip(entries, translated_instructions, translated_inputs, translated_outputs):
        translated = {
            "lang": lang,
            "instruction": ti,
            "input": tin,
            "output": tout,
            "original_instruction": orig.get("instruction", ""),
            "original_input": orig.get("input", ""),
            "original_output": orig.get("output", "")
        }
        for k, v in orig.items():
            if k not in ["instruction", "input", "output"]:
                translated[k] = v
        translated_entries.append(translated)

    return translated_entries

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_entries = [json.loads(line) for line in f]

    all_translations = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for lang in LANGS:
            for i in range(0, len(all_entries), BATCH_SIZE):
                batch = all_entries[i:i + BATCH_SIZE]
                futures.append(executor.submit(process_language, batch, lang))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
            all_translations.extend(future.result())

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_translations:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
