from .patch_model_2D import AutoPatchOverLapModel2D
from ..AFNONET.afnonet import AFNONet
import copy
import torch.nn as nn


class AFNONetPatch(AFNONet, AutoPatchOverLapModel2D):
    center_index_pool = {}
    around_index_pool = {}
    center_around_index_table = {}

    def __init__(self, config):
        assert config.patch_range
        config = copy.deepcopy(config)
        config.real_img_size = config.img_size
        patch_range = config.patch_range
        patch_range = self.good_tuple(patch_range, len(config.img_size))
        config.img_size = tuple(patch_range)
        self.patch_range = patch_range
        super().__init__(config)

    def build_PatchEmbedding(self, config):
        in_chans = config.in_chans
        embed_dim = config.embed_dim
        self.embedding = nn.Linear(in_chans, embed_dim, bias=False)

        def image_to_patches(x):
            x = x.permute(0, 2, 3, 1)
            x = self.embedding(x)
            x = x.permute(0, 3, 1, 2)
            x = self.image_to_patches(x)

            return x
        return image_to_patches

    def build_UpSampleBlock(self, config):
        embed_dim = config.embed_dim
        out_chans = config.out_chans
        self.projection = nn.Linear(embed_dim, out_chans, bias=False)

        def patches_to_image(x):
            x = self.patches_to_image(x)
            x = x.permute(0, 2, 3, 1)
            x = self.projection(x)
            x = x.permute(0, 3, 1, 2)
            #x = self.patches_to_image(x);print(x.shape)
            ##<--- In theory, you should put projection before patch-aggregating.
            ##However, since it is mean aggregation, thus post-projection is same as pre-projection .
            ##Then, use post projection is more efficient.
            return x
        return patches_to_image
