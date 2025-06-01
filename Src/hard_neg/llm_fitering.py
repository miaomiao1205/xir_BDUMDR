import json
import os
import random
import openai
from collections import Counter
from typing import List, Dict

class FalseNegativeFilter:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.prompt_template = """
        # Task Review: 
        Your task is to evaluate a Candidate Answer based on a given Question and Standard Answer. Use the following two evaluation criteria to          
        guide your assessment:
        # Evaluation Criteria

        ## Information Accuracy
        (1) Definition: Assess whether the Candidate Answer contains factual inaccuracies or misleading
        information. If a Standard Answer is provided, base your judgment on both the Question and the
        Standard Answer. If the Standard Answer is empty, evaluate based solely on the Question.
        (2) Scoring Guidelines:
            • 0: The Candidate Answer contains clear factual errors or significantly misrepresents the meaning.
            • 1: The Candidate Answer has minor inaccuracies, but the overall meaning is still mostly correct.
            • 2: The Candidate Answer is entirely accurate with no factual errors.

        ## Information Completeness
        (1) Definition: Evaluate how well the Candidate Answer addresses the key aspects of the Question.
        (2) Scoring Guidelines:
            • 0: Major aspects of the question are not addressed or key points are missing.
            • 1: Most key points are addressed, but some minor details are omitted.
            • 2: All major and minor points are fully addressed.

        # Input:
        Question: {Input_Question}
        Candidate Answer: {Input_Candidate_Answer}
        Standard Answer: {Input_Standard_Answer}

        # Output:
        """

    def generate_prompt(self, query: str, candidate_doc: str, positive_doc: str) -> str:
        return self.prompt_template.format(
            Input_Question=query,
            Input_Candidate_Answer=candidate_doc,
            Input_Standard_Answer=positive_doc
        )

    def get_llm_judgment(self, prompt: str, params: Dict) -> int:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=params["temperature"],
                top_p=params["top_p"],
                max_tokens=512
            )
            return int(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"API Error: {e}")
            return -1

    def get_majority_vote(self, query: str, candidate_doc: str, positive_doc: str) -> int:
        param_combinations = [
            {"top_p": 0.7, "temperature": 0.4},
            {"top_p": 0.8, "temperature": 0.6},
            {"top_p": 0.9, "temperature": 0.8},
            {"top_p": 0.8, "temperature": 0.4}
        ]

        scores = []
        prompt = self.generate_prompt(query, candidate_doc, positive_doc)

        for params in param_combinations:
            score = self.get_llm_judgment(prompt, params)
            if score != -1:
                scores.append(score)

        if not scores:
            return -1

        count = Counter(scores)
        majority_score, _ = count.most_common(1)[0]
        return majority_score

    def filter_negatives(self, query: str, candidates: List[str], positive_docs: List[str]) -> List[Dict]:
        valid_negatives = []
        positive_doc = random.choice(positive_docs) if positive_docs else ""

        for candidate in candidates:
            score = self.get_majority_vote(query, candidate, positive_doc)
            if score == 0:
                valid_negatives.append({
                    "query": query,
                    "candidate": candidate,
                    "standard_answer": positive_doc,
                    "score": score
                })
        return valid_negatives

    def process_file(self, input_path: str, output_dir: str, lang: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, f"{lang}_filtered.jsonl")
        with open(input_path, 'r', encoding='utf-8') as infile, \
                open(output_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                data = json.loads(line)
                query = data['query']
                query_id = data.get('query_id')
                candidates = [neg['text'] for neg in data.get('negative_passages', [])]
                positive_docs = [pos['text'] for pos in data.get('positive_passages', [])]

                filtered_negatives = self.filter_negatives(query, candidates, positive_docs)
                for neg in filtered_negatives:
                    neg['lang'] = lang
                    if query_id is not None: 
                        neg['query_id'] = query_id  
                    outfile.write(json.dumps(neg, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    filter = FalseNegativeFilter(api_key="sk-your-key-here")

    input_dir = "./Output/candidate"
    output_dir = "./Output/select"

    langs = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh']
    for lang in langs:
        input_path = os.path.join(input_dir, f"{lang}.jsonl")
        if os.path.exists(input_path):
            output_lang_dir = os.path.join(output_dir, lang)
            filter.process_file(input_path, output_lang_dir, lang)
        else:
            print(f"File not found: {input_path}")
