import json
import os
import faiss
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple
from tqdm import tqdm

class MultilingualRetriever:
    def __init__(self, model_paths: Dict[str, str], langs: List[str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.langs = langs
        self.models = {}
        self.tokenizers = {}
        
        # Load models for each language
        for model_name, path in model_paths.items():
            print(f"Loading {model_name} from {path}...")
            self.models[model_name] = AutoModel.from_pretrained(path).to(self.device)
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(path)
    
    def encode_batch(self, texts: List[str], model_name: str) -> np.ndarray:
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token

    def build_faiss_index(self, corpus_embeddings: np.ndarray) -> faiss.Index:
        """Build a FAISS index for efficient retrieval"""
        dim = corpus_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)
        faiss.normalize_L2(corpus_embeddings)  # Normalize for cosine similarity
        index.add(corpus_embeddings)
        return index

class LinguisticExpertSystem(nn.Module):
    def __init__(self, model_names: List[str], unified_dim: int = 512):
        super().__init__()
        # Load multiple multilingual retrieval models
        self.models = nn.ModuleDict({
            name: AutoModel.from_pretrained(name) 
            for name in model_names
        })
        
        # Initialize learnable linear transformation layers
        self.projection_layers = nn.ModuleDict({
            name: nn.Linear(model.config.hidden_size, unified_dim)
            for name, model in self.models.items()
        })
        
        # Attention weight calculation layer
        self.attention = nn.MultiheadAttention(unified_dim, num_heads=8)

    def encode(self, texts: List[str], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
        """Multi-model encoding"""
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        return {
            name: model(**inputs).last_hidden_state[:,0,:]  # Take CLS token
            for name, model in self.models.items()
        }

    def dynamic_fusion(self, query_reps: Dict[str, torch.Tensor], doc_reps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Dynamic feature fusion"""
        # Linear projection to unified dimension
        query_proj = [self.projection_layers[name](rep) for name, rep in query_reps.items()]
        doc_proj = [self.projection_layers[name](rep) for name, rep in doc_reps.items()]
        
        # Concatenate multi-model representations [num_models, batch_size, unified_dim]
        combined_query = torch.stack(query_proj, dim=0)
        combined_doc = torch.stack(doc_proj, dim=0)
        
        # Attention fusion (query as key, doc as value)
        attn_output, _ = self.attention(combined_query, combined_doc, combined_doc)
        return attn_output.mean(dim=0)  # Average to get final representation

class HardNegativeMiner:
    def __init__(self, model_paths: Dict[str, str], langs: List[str]):
        self.retriever = MultilingualRetriever(model_paths, langs)
        self.langs = langs
        # Initialize LinguisticExpertSystem with all configured models
        self.expert_system = LinguisticExpertSystem(
            model_names=list(model_paths.values()),
            unified_dim=768
        ).to(self.retriever.device)
    
    def process_language(self, lang: str):
        """Process queries and corpus for one language"""
        # Load queries
        query_path = f"./data/miracl/query/{lang}.jsonl"
        queries = []
        with open(query_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                queries.append((data['id'], data['query'], data.get('positives', [])))
        
        # Load corpus
        corpus_path = f"./data/miracl/corpus/{lang}.jsonl"
        corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                corpus.append((data['docid'], data['text']))
        
        print(f"Encoding {lang} corpus with LinguisticExpertSystem...")
        corpus_texts = [text for _, text in corpus]
        
        # Get multi-model representations for corpus
        corpus_reps_dict = self.expert_system.encode(
            corpus_texts, 
            self.retriever.tokenizers[next(iter(self.retriever.models))]
        )
        
        corpus_fused = self.expert_system.dynamic_fusion(corpus_reps_dict, corpus_reps_dict)
        corpus_fused = corpus_fused.cpu().numpy()
        
        index = self.retriever.build_faiss_index(corpus_fused)
        
        # Process queries
        output_path = f"./data/miracl/negatives/{lang}.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for qid, query, positives in tqdm(queries, desc=f"Processing {lang} queries"):
                # Encode query using all models
                query_reps_dict = self.expert_system.encode(
                    [query], 
                    self.retriever.tokenizers[next(iter(self.retriever.models))]
                )
                
                query_fused = self.expert_system.dynamic_fusion(query_reps_dict, corpus_reps_dict)
                query_fused = query_fused.cpu().numpy()
                faiss.normalize_L2(query_fused)
                
                # Search top 40 documents (excluding the query itself if present)
                _, indices = index.search(query_fused, k=41)
                
                # Filter out positive documents
                filtered_indices = [
                    idx for idx in indices[0][1:]  
                    if str(idx) not in positives  
                ]
                
                # Get negative documents
                negatives = []
                for idx in filtered_indices[:40]:  # Take top 40 after filtering
                    try:
                        negatives.append(corpus[idx][1])
                    except IndexError:
                        continue  
                
                # Save results including the positives field
                result = {
                    "id": qid,
                    "query": query,
                    "positives": positives,
                    "negatives": negatives[:40]  # Ensure exactly 40 negatives
                }
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def run(self):
        """Process all languages"""
        for lang in self.langs:
            self.process_language(lang)

if __name__ == "__main__":
    model_paths = {
        "BGE": "BAAI/bge-m3",
        "mE5": "intfloat/multilingual-e5-large"
    }
    langs = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh']
    
    # Run mining process
    miner = HardNegativeMiner(model_paths, langs)
    miner.run()