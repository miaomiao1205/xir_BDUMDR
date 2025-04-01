import os
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from .loss import WeightedContrastiveLoss  

import logging
logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


class TevatronTrainer(Trainer):
    def __init__(self, lang_weights, domain_weights, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.lang_weights = torch.tensor(lang_weights, device=self.args.device)
        self.domain_weights = torch.tensor(domain_weights, device=self.args.device)
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], 
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs:
            if isinstance(x, tuple):
                prepped = []
                for elem in x:
                    if isinstance(elem, torch.Tensor):
                        prepped.append(elem.to(self.args.device))
                    else:
                        prepped.append(super()._prepare_inputs(elem))
                prepared.append(tuple(prepped))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs):
        query, passage, lang_ids, domain_ids = inputs
        outputs = model(query=query, passage=passage)
        
        lang_w = self.lang_weights[lang_ids]
        domain_w = self.domain_weights[domain_ids]
        
        loss = outputs.loss * (lang_w + domain_w).mean()
        return loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    query = arg_val['query']
    passage = arg_val['passage']
    lang = arg_val['lang']
    domain = arg_val['domain']

    chunked = []
    for k in ['input_ids', 'attention_mask']:
        q_chunks = query[k].split(chunk_size, dim=0)
        p_chunks = passage[k].split(chunk_size, dim=0)
        l_chunks = lang.split(chunk_size, dim=0)
        d_chunks = domain.split(chunk_size, dim=0)
        
        for q, p, l, d in zip(q_chunks, p_chunks, l_chunks, d_chunks):
            chunked.append({
                arg_key: {
                    'query': {k: q},
                    'passage': {k: p},
                    'lang': l,
                    'domain': d
                }
            })

    return chunked


def get_dense_rep(x):
    if x.get('query') is not None:
        return x['query'].q_reps
    return x['passage'].p_reps


class GCTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer with Weighted Loss')
        if not _grad_cache_available:
            raise ValueError('Grad Cache package not available.')
            
        super(GCTrainer, self).__init__(*args, **kwargs)

        loss_fn = WeightedContrastiveLoss(
            language_weights=self.lang_weights,
            domain_weights=self.domain_weights,
            distributed=self.args.negatives_x_device,
            scale_loss=True
        )

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None,
            additional_input_keys=['lang', 'domain']
        )

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        queries, passages, langs, domains = self._prepare_inputs(inputs)
        inputs = {
            'query': queries,
            'passage': passages,
            'lang': langs,
            'domain': domains
        }

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        
        loss = self.gc(
            query_args=inputs['query'],
            doc_args=inputs['passage'],
            metadata={
                'languages': inputs['lang'],
                'domains': inputs['domain']
            },
            no_sync_except_last=_distributed
        )

        return loss / self._dist_loss_scale_factor