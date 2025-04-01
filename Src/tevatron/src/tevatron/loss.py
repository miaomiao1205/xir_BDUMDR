import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributed as dist

class WeightedContrastiveLoss:
    def __init__(self, 
                 language_weights: Tensor, 
                 domain_weights: Tensor,
                 n_target: int = 0, 
                 scale_loss: bool = True,
                 distributed: bool = False):

        self.language_weights = language_weights
        self.domain_weights = domain_weights
        self.n_target = n_target
        self.scale_loss = scale_loss
        self.distributed = distributed
        
        if distributed:
            assert dist.is_initialized(), "Distributed training has not been properly initialized."
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

    def __call__(self, 
                 x: Tensor, 
                 y: Tensor, 
                 languages: Tensor, 
                 domains: Tensor,
                 target: Tensor = None, 
                 reduction: str = 'mean'):

        if self.distributed:
            # Gather tensors from all processes
            x = self.gather_tensor(x)
            y = self.gather_tensor(y)
            languages = self.gather_tensor(languages)
            domains = self.gather_tensor(domains)
        
        batch_size = x.size(0)
        if target is None:
            target_per_qry = y.size(0) // batch_size
            target = torch.arange(
                0, batch_size * target_per_qry, target_per_qry, 
                device=x.device, dtype=torch.long
            )
        
        # Compute logits and base loss
        logits = torch.matmul(x, y.transpose(0, 1))
        base_loss = F.cross_entropy(logits, target, reduction='none')
        
        # Compute weights for negative samples
        neg_mask = torch.ones_like(logits, dtype=torch.bool)
        neg_mask[torch.arange(batch_size), target] = False
        
        # Get language and domain weights for negative samples
        neg_languages = languages.view(-1, 1).expand(-1, y.size(0))[neg_mask]
        neg_domains = domains.view(1, -1).expand(batch_size, -1)[neg_mask]
        
        language_w = self.language_weights[neg_languages]
        domain_w = self.domain_weights[neg_domains]
        
        weights = language_w + domain_w
        
        # Normalize weights per query
        weights = weights.view(batch_size, -1)
        weights = weights / weights.sum(dim=1, keepdim=True)
        weights = weights.view(-1)
        
        # Apply weights to loss
        weighted_loss = base_loss * weights[target]
        
        if reduction == 'mean':
            weighted_loss = weighted_loss.mean()
            if self.distributed and self.scale_loss:
                weighted_loss = weighted_loss * self.world_size
        elif reduction == 'sum':
            weighted_loss = weighted_loss.sum()
            
        return weighted_loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)