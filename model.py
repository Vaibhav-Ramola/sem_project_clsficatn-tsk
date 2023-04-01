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
