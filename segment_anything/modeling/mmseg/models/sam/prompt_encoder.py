import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Type
import math
from .common import MLPBlock

batch_size = 1

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Adapter(nn.Module):
    def __init__(self, input_dim, mid_dim):
        super().__init__()
        self.model = MLP(
            input_dim=input_dim, hidden_dim=mid_dim, output_dim=input_dim, num_layers=2
        )

    def forward(self, features):
        out = features + self.model(features)
        return out


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class TwoWayTransformerVisualSampler(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    batch_size=batch_size,
                )
            )

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_coord,
    ) -> Tuple[Tensor, Tensor]:
        point_embedding = F.grid_sample(image_embedding, point_coord, align_corners=False).squeeze(2)
        point_pe = F.grid_sample(image_pe, point_coord, align_corners=False).squeeze(2)
        point_pe = point_pe.permute(0, 2, 1)
        point_embedding = point_embedding.permute(0, 2, 1)
        original_shape = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        for layer in self.layers:
            image_embedding, point_embedding = layer(
                image_embedding,
                point_embedding,
                image_pe,
                point_pe,
            )
        return image_embedding


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.global_query = nn.parameter.Parameter(data=0.1 * torch.randn(batch_size, 10, embedding_dim))

    def forward(self, img_embed, point_embed, img_pe, point_pe) -> Tuple[Tensor, Tensor]:
        q = torch.cat([self.global_query, point_embed], dim=1)
        
        self_out = self.self_attn(q=q, k=q, v=q)
        self_out = self.norm1(self_out)

        queries = q + self_out
        queries = self.norm2(queries)
        point_embed = queries[:, 10:, :]
        queries = queries[:, :10, :]

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        attn_out = self.cross_attn_image_to_token(q=img_embed, k=queries, v=queries)
        keys = img_embed + attn_out
        keys = self.norm4(keys)

        return keys, point_embed


class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head


    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    

class PromptEncoder(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        num_pos_feats: int = 128,
        mask_prompt = False
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
             torch.randn((2, num_pos_feats)),
        )
        self.mask_prompt = mask_prompt
        if mask_prompt:
            self.default_prompt = nn.parameter.Parameter(torch.randn(1, 256, 32, 32))
            self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 256 // 4, kernel_size=3, stride=3),
            LayerNorm2d(256 // 4),
            nn.GELU(),
            nn.Conv2d(256 // 4, 256, kernel_size=3, padding = 1, stride=1),
            LayerNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coord,
        masks = None,
        img_size = [1024, 1024],
        feat_size = [64, 64]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_pe = self.get_img_pe(feat_size, device=image_embeddings.device).detach()
        point_coord[:, :, 0] = (point_coord[:, :, 0]+0.5) * 2 / img_size[1] - 1
        point_coord[:, :, 1] = (point_coord[:, :, 1]+0.5) * 2 / img_size[0] - 1
        
        global batch_size
        batch_size = image_embeddings.size()[0]
        point_coord = point_coord.reshape(batch_size,1,-1,2)
        image_pe = image_pe.repeat(batch_size, 1, 1, 1)
        features = self.transformer(image_embeddings, image_pe, point_coord)
        #print("before reshape ", features.size())
        features = features.transpose(1,2).reshape([batch_size, -1] + feat_size)
        #print("after reshape ", features.size())
        return features

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  

    def get_img_pe(self, size: Tuple[int, int], device) -> torch.Tensor:
        h, w = size
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1).unsqueeze(0)  # C x H x W
