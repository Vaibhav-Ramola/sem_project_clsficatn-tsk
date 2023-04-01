'''
Implementation Reference : 
1. Paper : 'Attention is all you need' by Vaswani
2. Medium Article by Alessandro Lamberti, titled : 'ViT — VisionTransformer, a Pytorch implementation'

'''


from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torchvision.transforms import Compose, Resize, ToTensor
from torchsummary import summary
from torch import Tensor, nn
import torch.nn.functional as F
import torch
from PIL import Image
import cv2  
import matplotlib.pyplot as plt


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 5, embed_size: int = 786, in_size: int = 32, num_patches: int = 4) -> None:
        super(PatchEmbedding, self).__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.in_size = in_size
        self.num_patches = num_patches
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_size, num_patches, stride=(num_patches, num_patches)),
            Rearrange('b e (w) (h) -> b (w h) e')
            #-> rearranges the output vector from the conv layer
            # for example o/p from conv : (1 786 8 8)
            # then o/p from Rearrange is  : (1 64 786)
        )
        self.cls_token = nn.Parameter(torch.randn(1, embed_size))
        self.position_embedding = nn.Parameter(torch.randn((in_size // num_patches)**2+1, embed_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        out = self.embed(x)
        cls_token = repeat(self.cls_token, '() s e -> b s e', b = b)
        out = torch.cat([cls_token, out], dim=1) 
        # prepending the cls token and output tensor along dim-1
        out += self.position_embedding
        # the sum propagates to all batches i.e. dim=0
        return out
    
class MSA(nn.Module):
    '''
    1. Add dropout
    2. 
    '''
    def __init__(self, embed_size: int = 786, num_heads: int = 8) -> None:
        super(MSA, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_size, 3 * embed_size)
        self.rearg = Rearrange('b n (h qkv e) -> b qkv h n e', h=num_heads, qkv=3)
        self.linear = nn.Linear(self.embed_size, self.embed_size)
    
    def forward(self, x):
        qkv = self.qkv(x)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        attention = torch.einsum('bhqd bhkd -> bhqk', queries, keys)
        softmax_attention = F.softmax(attention, dim=-1) // (self.embed_size ** (1/2))
        # dim = -1 means apply softmax along the last dimension i.e. along the columns
        context = torch.einsum('bhql bhld -> bhqd', softmax_attention, values)
        context = rearrange(context, 'b h v d -> b v (h d)')
        out = self.linear(context)

        return out
    
class FeedFwd(nn.Module):
    def __init__(self, embed_size: int = 786, z: int = 5, dropout: float = 0.2) -> None:
        super(FeedFwd, self).__init__()
        self.embed_size = embed_size
        self.z = z
        self.feedFwd = nn.Sequential(
            nn.Linear(embed_size, z * embed_size),
            nn.BatchNorm1d(z * embed_size),
            # Should batchnorm be appllied here as we'll take layer norm later ?
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(z * embed_size, embed_size),
        )
    
    def forward(self, x):
         return self.feedFwd(x)
    

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size: int = 786, ff_dropout: float = 0.2, z: int = 5, num_heads: int = 8, num_patches: int = 4, in_size: int = 32) -> None:
        super(TransformerEncoder, self).__init__()
        self.msa = MSA(embed_size=embed_size, num_heads=num_heads)
        self.feedForward = FeedFwd(embed_size=embed_size, dropout=ff_dropout, z=z)
        self.ln1 = nn.LayerNorm(embed_size)
        # Do I have to use a different initialized layer norm object 
        # or can I use the previous one ? 
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        out = self.msa(x)
        out = self.ln1(out+x)
        x = out
        out = self.feedForward(out)
        out = self.ln2(x+out)

        return out
    
class VIT(nn.Module):
    def __init__(self, 
                 embed_size: int = 786, 
                 ff_dropout: float = 0.2, 
                 z: int = 5, 
                 num_heads: int = 8, 
                 num_patches: int = 4, 
                 in_size: int = 32, 
                 L: int = 12, 
                 num_classes: int = 3, 
                 in_channels: int = 5) -> None:
        super(VIT, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, embed_size=embed_size, in_size=in_size, num_patches=num_patches)
        self.transformer_encoders = nn.Sequential()

        for i in range(L):
            self.transformer_encoders.append(TransformerEncoder(embed_size=embed_size, ff_dropout=ff_dropout, in_size=in_size, num_heads=num_heads, num_patches=num_patches, z=z))

        self.classificationHead = nn.Sequential(
            Reduce('b n e -> b e', 'mean'), # takes the mean along the 1st dimension
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes),     
            # Can an additional hidden layer be added here ?
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoders(x)
        x = self.classificationHead(x)

        return x
    

