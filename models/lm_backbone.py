from torch import nn

from .transformer import ContinuousTransformer


class AudioLMBackbone(nn.Module):
    def __init__(self, embed_dim: int, use_generation_cache=False, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_generation_cache = use_generation_cache
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ContinuousTransformerAudioLMBackbone(AudioLMBackbone):
    def __init__(self, embed_dim: int, cross_attn_cond_dim: int = 0, prepend_cond_dim: int = 0, project_cross_attn_cond: bool = False, **kwargs):
        super().__init__(embed_dim=embed_dim)
        self.model = ContinuousTransformer(
            dim=embed_dim,
            dim_in=embed_dim,
            dim_out=embed_dim,
            cross_attend = cross_attn_cond_dim > 0,
            cond_token_dim = embed_dim if project_cross_attn_cond else cross_attn_cond_dim,
            causal=True,
            **kwargs
        )
        if prepend_cond_dim > 0:
            self.to_prepend_embed = nn.Sequential(nn.Linear(prepend_cond_dim, embed_dim, bias=False), nn.SiLU(), nn.Linear(embed_dim, embed_dim, bias=False))
        else:
            self.to_prepend_embed = nn.Identity()
        if cross_attn_cond_dim > 0 and project_cross_attn_cond:
            self.to_cross_attn_embed = nn.Sequential(nn.Linear(cross_attn_cond_dim, embed_dim, bias=False), nn.SiLU(), nn.Linear(embed_dim, embed_dim, bias=False))
        else:
            self.to_cross_attn_embed = nn.Identity()

    def forward(self, x, mask=None, prepend_cond=None, prepend_cond_mask=None, cross_attn_cond=None, global_cond=None, use_cache=False):
        prepend_length = 0
        if not isinstance(self.to_prepend_embed, nn.Identity) and prepend_cond is not None:
            prepend_cond = self.to_prepend_embed(prepend_cond)
            prepend_length = prepend_cond.shape[1]
            if prepend_cond_mask is not None:
                prepend_cond_mask = prepend_cond_mask.bool()
        if not isinstance(self.to_cross_attn_embed, nn.Identity) and cross_attn_cond is not None:
            cross_attn_cond = self.to_cross_attn_embed(cross_attn_cond)
        return self.model(x, mask=mask, context=cross_attn_cond, prepend_embeds=prepend_cond, prepend_mask=prepend_cond_mask)[:, prepend_length:, :]


