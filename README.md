<!--
<div align="center">
  <img src="figs/logo.png" alt="DMDR Logo" width="500">
</div>


# A Data-Driven Framework for Multilingual Dense Retrieval

This repository contains the code and the generated hard negative samples for the paper "A Data-Driven Framework for Multilingual Dense Retrieval".

# 📄 Abstract
Multilingual dense retrieval aims to retrieve relevant documents across multiple languages. The challenge lies in aligning representations of different languages in a shared vector space.The common practice is to fine-tune the dense retriever via contrastive learning, whose effectiveness highly relies on the quality of the negative sample and the construction of mini-batch data. In this study, we propose a data-driven framework DMDR for multilingual dense retrieval fine-tuning by obtaining high-quality hard negative samples and effective mini-batch data and integrating the negative sampling weight with the contrastive learning objective. The extensive experimental results on a multilingual retrieval benchmark MIRACL demonstrate the effectiveness of our proposed DMDR by outperforming several existing strong baselines.

# 🤩 The Framework of DMDR
<div align="center">
  <img src="figs/DMDR.png" alt="DMDR ">
</div>

Our DMDR framework including three stages: i) construction of hard negative set, ii) LLM-aided hard negative generation, and iii) effective mini-batch construction to facilitate contrastive learning.
-->

# 👉 Quick Start
# Table of Contents:
* [Environment](#environment)
* [Data and Models Preparation](#data-and-models-preparation)
* [Negative samples construction and False negative samples filtering](#negative-samples-construction-and-false-negative-samples-filtering)
* [LLM-aided hard negative samples generation](#llm-aided-hard-negative-samples-generation)
* [Train and Evaluation](#train-and-evaluation)

## 1. Environment <a name="environment"></a>
Follow the commands below to establish a plausible environment.
```bash
conda create --name dmdr python=3.7
conda activate dmdr
pip install -r requirements.txt
```

## 2. Data and Models Preparation <a name="data-and-models-preparation"></a>

For public datasets and checkpoints can be download from [MIRACL](https://huggingface.co/datasets/miracl/miracl-corpus)、[Alpaca](https://github.com/tatsu-lab/stanford_alpaca)、[mDPR](https://huggingface.co/castorini/mdpr-tied-pft-msmarco)、[mE5<sub>large</sub>](https://huggingface.co/intfloat/multilingual-e5-large) and [BGE](https://huggingface.co/BAAI/bge-m3). And put them in the Data and PLM folders.

## 3. Negative samples construction and False negative samples filtering <a name="negative-samples-construction-and-false-negative-samples-filtering"></a>

Run the following two python scripts for hard negative candidate set construction and false negative sample filtering([GPT-4o](https://platform.openai.com/docs/models/gpt-4o)), respectively.

```bash
cd Src
python hard_neg/candidate_generation.py
python hard_neg/llm_fitering.py
```

## 4. LLM-aided Hard Negative Generation <a name="llm-aided-hard-negative-samples-generation"></a>
At this stage, we finetune LLaMA-3.1-70B with the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) tool to generate difficult negative samples.

>**LLaMA-Factory installation**

Run the following command to quickly install LLaMA-Factory.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

>**Multilingual Instruction Fine-tuning**

First, we use multilingual version of the Alpaca dataset obstained by [Google Translate](https://cloud.google.com/translate/docs/basic/translating-text) for multilingual instruction fine-tuning. The multilingual version of the Alpaca dataset is obtained by running the following script.

```bash
cd Src
python LLM_generation/translate.py
```

​Then, replace the yaml file in the LLM_generation folder with the corresponding file(examples/train_lora/llama3_lora_sft_ds3.yaml、examples/merge_lora/llama3_lora_sft.yaml) in the LLaMA-Factory. And run the following two commands for LoRA fine-tuning and model merging.

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

>**Hard Negative Samples Generation**

Use the following two scripts to generate the summary, the query, and the hard negative samples in turn.

```bash
cd Src
python LLM_generation/inference.py
python hard_neg/candidate_generation.py
```

## 5. Train and EvaluationEnvironment <a name="train-and-evaluation"></a>

>**Tevatron installation**
We use the [tevatron](https://github.com/texttron/tevatron/tree/tevatron-v1) tool for DMDR training. Run the following command for a quick tevatron installation.

```bash
cd src/tevatron
pip install --editable .
```

>**Train**

We train on a machine with 2xH100 GPU, if the GPU resources are limited for you, please train with gradient cache. To train DMDR, please run the following commands.

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --master_port 22345 --nproc_per_node=2 -m tevatron.driver.train \
  --output_dir xxx \
  --do_train \
  --model_name_or_path xxx \
  --dataset_name Tevatron/msmarco-passage \
  --data_cache_dir ./msmarco-passage-train-cache \
  --train_dir xxx \
  --save_steps 10000 \
  --q_max_len 64 \
  --p_max_len 256 \
  --fp16 \
  --train_n_passages 8 \
  --learning_rate 3e-6 \
  --num_train_epochs 16 \
  --per_device_train_batch_size 12 \
  --overwrite_output_dir \
  --dataloader_num_workers 4 \
  --negatives_x_device > 
```

The query and document lengths are set to 64 and 256, train_n_passages is set to 8, which means that one positive and seven negative samples will be used for each query during training, bacth_size is set to 12, and epochs are set to 16.

>**Encoding**

After training, we need to encode the query and the document separately. To encode query and corpus, use the following two commands, respectively.

```bash
# encoding query

#!/bin/bash

ID=$1
LANG=$2
PARAM=$3

CUDA_VISIBLE_DEVICES=${ID} nohup python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path xxx \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --data_cache_dir ./msmarco-passage-train-cache \
  --dataset_name Tevatron/msmarco-passage \
  --encode_in_path ./DATA/miracl/query/${LANG}.jsonl \
  --encoded_save_path ${LANG}_${PARAM}.pkl \
  --q_max_len 64 \
  --encode_is_qry > 
```

Encoding corpus

```bash
# encoding corpus
#!/bin/bash

ID=$1
LANG=$2
PARAM=$3

for s in $(seq -f "%02g" 0 5)
do
  CUDA_VISIBLE_DEVICES=${ID} nohup python -m tevatron.driver.encode \
    --output_dir=temp \
    --model_name_or_path ./model/mma_model_${PARAM} \
    --fp16 \
    --per_device_eval_batch_size 156 \
    --dataset_name Tevatron/msmarco-passage-corpus \
    --data_cache_dir ./msmarco-passage-train-cache \
    --p_max_len 256 \
    --encode_in_path ./DATA/miracl/corpus/${LANG}.jsonl \
    --encoded_save_path ${LANG}_${PARAM}_${s}.pkl \
    --encode_num_shard 6 \
    --encode_shard_index ${s}
done
```

>**Dense Retrieval**

After encoding the query and document, vector retrieval is performed using faiss on the two resulting .pkl files by running the following command.

```bash
#!/bin/bash

LANG=$1
PARAM=$2

nohup python -m tevatron.faiss_retriever \
  --query_reps ${LANG}/${LANG}_${PARAM}.pkl \
  --passage_reps ${LANG}/"${LANG}_${PARAM}*.pkl" \
  --depth 100 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to xxx
```


>**Evaluation**

Finally, use the pyserini tool to evaluate the retrieval performance by running the following command.

  ```bash
  #!/bin/bash
  
  LANG=$1
  PARAM=$2
  
  python -m tevatron.utils.format.convert_result_to_trec \
    --input ${LANG}_${PARAM}.txt \
    --output ${LANG}_${PARAM}.trec
  
  python -m pyserini.eval.trec_eval \
      -m recall.100 -m ndcg_cut.10 \
      ${LANG}.tsv ${LANG}_${PARAM}.trec
  ```

# License
This repository is licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

