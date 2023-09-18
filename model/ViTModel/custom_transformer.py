import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit_pytorch.vit import  PreNorm, FeedForward

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1) 
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (b h n d) (b h d n) -> (b h n n)
        if attn_mask is not None:
            dots.masked_fill_(attn_mask, -torch.inf) # mask = 1 mean do mask this part

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Flow_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.dim = dim
        self.num_heads = self.heads = heads
        self.qkv       = nn.Linear(dim, inner_dim * 3, bias=False)
        self.proj      = nn.Linear(inner_dim, dim)
        self.softmax   = nn.Softmax(dim=-1)
        self.scale     = dim_head ** -0.5

    def kernel(self, x):
        x = torch.sigmoid(x)
        return x

    def my_sum(self, a, b):
        # "nhld,nhd->nhl"
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, x, attn_mask=None):
        assert attn_mask is None, "temporarily not work"
        B_, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim = -1) 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # kernel
        q, k = self.kernel(q), self.kernel(k)
        v    = v/(self.scale*C // self.num_heads) # to avoid explore for float32  # the old version is set 40
        # normalizer
        sink_incoming = 1.0 / (self.my_sum(q + 1e-6, k.sum(dim=2) + 1e-6) + 1e-6)
        source_outgoing = 1.0 / (self.my_sum(k + 1e-6, q.sum(dim=2) + 1e-6) + 1e-6)
        conserved_sink = self.my_sum(q + 1e-6, (k * source_outgoing[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = self.my_sum(k + 1e-6, (q * sink_incoming[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
        # competition
        source_competition = torch.softmax(conserved_source, dim=-1) * float(k.shape[2])
        # multiply
        kv = k.transpose(-2, -1) @ (v * source_competition[:, :, :, None])
        x_update = ((q @ kv) * sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]
        x = rearrange(x_update, 'b h n d -> b n (h d)')
        #x = (x_update).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class TransformerBase(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, AttentionType=Attention, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, AttentionType(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, attn_mask=attn_mask) + x
            x = ff(x) + x
        return x

class Transformer(TransformerBase):
    def __init__(self, *args,**kargs):
        super().__init__( *args, AttentionType=Attention, **kargs)

class Flowformer(TransformerBase):
    def __init__(self, *args,**kargs):
        super().__init__( *args, AttentionType=Flow_Attention, **kargs)

class ViT3D(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, FormerType=Transformer,use_time_embedding=False,  channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_z, image_height, image_width = self.triple(image_size)
        patch_z, patch_height, patch_width = self.triple(patch_size)
        self.image_z = image_z
        self.image_h = image_height
        self.image_w = image_width
        self.num_classes = num_classes
        self.patch_z = patch_z
        self.patch_h = patch_height
        self.patch_w = patch_width

        
        assert image_z%patch_z == 0 and image_height % patch_height == 0 and image_width % patch_width == 0, f'Image dimensions ({image_z},{image_height},{image_width}) must be divisible by the patch size ({patch_z},{patch_height},{patch_width}).'

        self.inner_z = image_z//patch_z
        self.inner_h = image_height // patch_height
        self.inner_w = image_width // patch_width

        num_patches = (image_z//patch_z) * (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_z* patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (z p0) (h p1) (w p2) -> b (z h w) (p0 p1 p2 c)', p0 = patch_z, p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.use_time_embedding = use_time_embedding
        if self.use_time_embedding:self.time_embedding = nn.Linear(4, dim)
        self.dropout       = nn.Dropout(emb_dropout)
        self.transformer   = FormerType(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        out_dim = num_classes*patch_z* patch_height * patch_width
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
    
    def triple(self,t):
        return t if isinstance(t, tuple) else (t, t, t )

    def forward(self, img, time_stamp=None, attn_mask=None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        if self.use_time_embedding:
            time_embedding = self.time_embedding(time_stamp).unsqueeze(1)            
            x = x + time_embedding #(B,T,L) + (B,1,L) -> (B,T,L)
        #x += self.pos_embedding[:, :(n + 1)] #remove time
        x = self.dropout(x)
        x = self.transformer(x,attn_mask)
        x = self.mlp_head(x)
        x = rearrange(x,"b (z h w) (p0 p1 p2 c) -> b c (z p0) (h p1) (w p2)", c=self.num_classes, 
                    p0=self.patch_z,p1=self.patch_h,p2=self.patch_w, z=self.inner_z, h =self.inner_h, w = self.inner_w)
        return x

class ViT3DTransformer(ViT3D):
    def __init__(self, *args,**kargs):
        super().__init__( *args, FormerType=Transformer, **kargs)

class ViT3DFlowformer(ViT3D):
    def __init__(self, *args,**kargs):
        super().__init__( *args, FormerType=Flowformer, **kargs)
