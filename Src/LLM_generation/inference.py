import json
import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "output/llama3_lora_sft"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device)

langs = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh']

def generate_text(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[-1]
    
    try:
        outputs = model.generate(
            **inputs,
            max_length=input_length + max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Error during generation: {e}")
        return ""

def process_language(lang):
    input_file = f"{lang}.jsonl"
    output_file = f"{lang}_generate.jsonl"
    
    results = []
    
    with jsonlines.open(input_file) as reader:
        for item in tqdm(reader, desc=f"Processing {lang}"):
            if "positives" not in item:
                continue
                
            summary_prompt = (
                "You are a skilled summarizer. Please summarize the following text into a concise and informative summary "
                "that captures the main points in one or two sentences:\n\n"
                f"{item['positives']}\n\nSummary:"
            )
            summary = generate_text(summary_prompt)
            
            query_prompt = (
                "You are an expert search query generator. Based on the summary provided, please generate a search query "
                "that effectively captures the key aspects of the summary. The query should be concise and specific:\n\n"
                f"{summary}\n\nQuery:"
            )
            query = generate_text(query_prompt)
            
            result = {
                "original_text": item['positives'],
                "summary": summary,
                "generated_query": query
            }
            results.append(result)
    
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(results)

if __name__ == "__main__":
    for lang in langs:
        try:
            print(f"Processing language: {lang}")
            process_language(lang)
        except Exception as e:
            print(f"Error processing {lang}: {e}")