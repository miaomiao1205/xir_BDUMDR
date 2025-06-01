import torch
import torch.nn as nn
import os
from transformers import AutoModel
from typing import Dict, List, Optional


class MultilingualFusionModel(nn.Module):
    """A fusion model for multilingual pretrained language models.
    
    Args:
        model_configs: Dictionary mapping model names to HuggingFace paths
        unified_dim: Dimension for unified representation space (default: 768)
        device: Target device
    """

    def __init__(
        self,
        model_configs: Dict[str, str],
        unified_dim: int = 768,
        device: Optional[str] = None
    ):
        super().__init__()
        self.model_configs = model_configs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize all sub-models
        self.models = nn.ModuleDict()
        for name, path in model_configs.items():
            model = AutoModel.from_pretrained(path)
            if self.device == "cuda":
                model = model.to(self.device)
            self.models[name] = model
        
        # 2. Dynamic projection layers
        self.projections = nn.ModuleDict()
        for name, model in self.models.items():
            hidden_size = model.config.hidden_size
            proj = nn.Linear(hidden_size, unified_dim).to(self.device)
            init.xavier_uniform_(proj.weight)
            init.zeros_(proj.bias)
            self.projections[name] = proj
        
        # 3. Fusion modules
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=unified_dim,
            num_heads=8,
            batch_first=True
        ).to(self.device)
        self.layer_norm = nn.LayerNorm(unified_dim).to(self.device)

    def get_individual_outputs(
        self,
        inputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Extract raw outputs from each model with input handling.
        
        Args:
            inputs: Dictionary of model inputs for each sub-model
            
        Returns:
            Dictionary of model outputs
        """
        outputs = {}
        for name, model_inputs in inputs.items():
            # Ensure inputs are on correct device
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            model_output = self.models[name](**model_inputs)
            
            # Generic sentence representation extraction
            if hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
                outputs[name] = model_output.pooler_output
            else:
                # Default to CLS token
                outputs[name] = model_output.last_hidden_state[:, 0, :]
        return outputs

    def fuse_representations(
        self,
        model_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Fuse multiple model representations.
        
        Args:
            model_outputs: Dictionary of model outputs
            
        Returns:
            Fused representation tensor
        """
        projected = []
        for name, output in model_outputs.items():
            projected.append(self.projections[name](output))
        
        combined = torch.stack(projected, dim=1)
        attn_output, _ = self.fusion_attn(combined, combined, combined)
        
        fused = self.layer_norm(combined + attn_output)
        return fused.mean(dim=1)

    def forward(
        self,
        inputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Forward pass through the fusion model.
        
        Args:
            inputs: Dictionary of model inputs
            
        Returns:
            Fused output tensor
        """
        inputs = {name: {k: v.to(self.device) for k, v in model_inputs.items()} 
                 for name, model_inputs in inputs.items()}
        outputs = self.get_individual_outputs(inputs)
        return self.fuse_representations(outputs)


def save_model(
    model: MultilingualFusionModel,
    output_path: str
) -> None:
    """Save model with state dict and configuration.
    
    Args:
        model: Model instance to save
        output_path: Path to save the model
    """
    save_data = {
        'state_dict': model.state_dict(),
        'model_configs': model.model_configs,
        'unified_dim': model.layer_norm.normalized_shape[0]
    }
    torch.save(save_data, output_path)


if __name__ == "__main__":
    MODEL_CONFIG = {
        "BGE": "BAAI/bge-m3",
        "mE5": "intfloat/multilingual-e5-large",
        # Add more models as needed
    }
    
    fusion_model = MultilingualFusionModel(MODEL_CONFIG)
    os.makedirs("./PLM/fusion_model", exist_ok=True)
    save_model(fusion_model, "./PLM/fusion_model/fusion_model.bin")
