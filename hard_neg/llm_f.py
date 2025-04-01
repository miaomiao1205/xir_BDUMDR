import json
import os
import random
from collections import Counter
from typing import List, Dict, Tuple

import openai

class FalseNegativeFilter:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.prompt_template = """# Task Review:

Your task is to evaluate the {Standard Answer} and {Candidate Answer} according to the {Evaluation Criteria}.  
You will receive a {Question}, a {Standard Answer}, and a {Candidate Answer}. Based on the assessment results for each criterion, the final score is determined as follows: If there is a score of 0 in any criterion, the final output is 0. If there is no 0 score, count the number of 1 and 2 scores. If there are more 1 score than 2 scores, output 1. Otherwise, if the number of 2 scores is equal to or greater than the number of 1 scores, output 2.

## Evaluation Criteria

### -Information Accuracy
**(2) Scoring:** 0: Contains clear factual errors. 1: Minor deviations. 2: Fully accurate.

### -Information Completeness
**(2) Scoring:** 0: Misses key aspects. 1: Covers most points. 2: Fully comprehensive.

[Question]: {Question}
(Candidate Answer): {Candidate Answer}
{Standard Answer}: {Standard Answer}

{Output}"""

    def generate_prompt(self, query: str, candidate_doc: str, positive_doc: str) -> str:
        return self.prompt_template.format(
            Question=query,
            Candidate_Answer=candidate_doc,
            Standard_Answer=positive_doc,
            Evaluation_Criteria="Evaluation Criteria"
        )

    def get_llm_judgment(self, prompt: str, params: Dict) -> int:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=params["temperature"],
                top_p=params["top_p"],
                max_tokens=10
            )
            return int(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"API Error: {e}")
            return -1  # Mark error situation

    def get_majority_vote(self, query: str, candidate_doc: str, positive_doc: str) -> int:
        param_combinations = [
            {"top_p": 0.7, "temperature": 0.4},
            {"top_p": 0.8, "temperature": 0.6},
            {"top_p": 0.9, "temperature": 0.8},
            {"top_p": 0.8, "temperature": 0.4}  # Fourth combination
        ]
        
        scores = []
        prompt = self.generate_prompt(query, candidate_doc, positive_doc)
        
        for params in param_combinations:
            score = self.get_llm_judgment(prompt, params)
            if score != -1:  # Ignore error results
                scores.append(score)
        
        if not scores:
            return -1  # All calls failed
        
        # Majority vote
        count = Counter(scores)
        majority_score, _ = count.most_common(1)[0]
        return majority_score

    def filter_negatives(self, 
                        query: str, 
                        candidates: List[str], 
                        positive_docs: List[str]) -> List[str]:
        valid_negatives = []
        positive_doc = random.choice(positive_docs) if positive_docs else ""
        
        for candidate in candidates:
            score = self.get_majority_vote(query, candidate, positive_doc)
            if score == 0:
                valid_negatives.append(candidate)
        
        return valid_negatives

    def process_file(self, input_path: str, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(os.path.join(output_dir, 'prompts.txt'), 'w', encoding='utf-8') as outfile:
            for line in infile:
                data = json.loads(line)
                query = data['query']
                candidates = data['negatives']
                positive_docs = data['positives']
                
                if positive_docs:
                    positive_doc = random.choice(positive_docs)
                else:
                    positive_doc = ""
                
                for candidate in candidates:
                    prompt = self.generate_prompt(query, candidate, positive_doc)
                    outfile.write(prompt + '\n\n')

if __name__ == "__main__":
    # replace with your actual API key
    filter = FalseNegativeFilter(api_key="sk-your-key-here")
    
    
    input_dir = "/Data/miracl/negatives"
    output_dir = "/Data/miracl/filtered_negatives"
    
    # Process each language file
    langs = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh']
    for lang in langs:
        input_path = os.path.join(input_dir, f"{lang}.jsonl")
        if os.path.exists(input_path):
            output_lang_dir = os.path.join(output_dir, lang)
            filter.process_file(input_path, output_lang_dir)
        else:
            print(f"File not found: {input_path}")