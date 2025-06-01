import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "output/llama3_lora_sft"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device)
vocab_size = model.config.vocab_size

langs = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh']

lang_names = {
    'ar': 'Arabic',
    'bn': 'Bengali',
    'en': 'English',
    'es': 'Spanish',
    'fa': 'Persian',
    'fi': 'Finnish',
    'fr': 'French',
    'hi': 'Hindi',
    'id': 'Indonesian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ru': 'Russian',
    'sw': 'Swahili',
    'te': 'Telugu',
    'th': 'Thai',
    'zh': 'Chinese'
}

def format_prompt(role, content):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

def generate_text(user_input, max_new_tokens=256):
    input_text = "<|begin_of_text|>" + format_prompt("user", user_input) + format_prompt("assistant", "")
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"]

    if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
        offending_ids = input_ids[(input_ids >= vocab_size) | (input_ids < 0)]
        print(f"Invalid token IDs detected: {offending_ids}")
        return ""

    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = input_ids.shape[-1]

    try:
        outputs = model.generate(
            **inputs,
            max_length=input_length + max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            early_stopping=True
        )
        return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Error during generation: {e}")
        return ""

def process_language(lang):
    input_file = f"./Output/expand/{lang}_expand.jsonl"
    output_file = f"./Output/generate/summary/{lang}.jsonl"
    results = []

    language_name = lang_names.get(lang, lang)

    with jsonlines.open(input_file) as reader:
        for item in tqdm(reader, desc=f"Generating summary for {lang}"):
            if "standard_answer" not in item:
                continue

            summary_prompt = (
                f"The following text is in {language_name}. Please summarize it in the same language. "
                f"Summarize the content into one or two concise and informative sentences:\n\n"
                f"{item['standard_answer']}\n\nSummary:"
            )
            summary = generate_text(summary_prompt)
            item["summary"] = summary
            results.append(item)

    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(results)

if __name__ == "__main__":
    for lang in langs:
        try:
            print(f"Processing language: {lang}")
            process_language(lang)
        except Exception as e:
            print(f"Error processing {lang}: {e}")
