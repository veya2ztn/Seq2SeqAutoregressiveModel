import torch
import torch.nn as nn
from networks.Codebook import Codebook, Codebook_EAMupdate
from networks.Decoder import Decoder
from networks.Encoder import Encoder




class VQVAE(nn.Module):
    def __init__(self, img_channel, latent_dim=256, num_codebook_vectors=2048, beta=0.25, ch=64, resolution=32, \
                num_res_blocks=3, ch_mult=(1,1,2,2,4), attn_resolutions = [8]):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(img_channel, latent_dim, ch, resolution, num_res_blocks, ch_mult, attn_resolutions)
        self.decoder = Decoder(img_channel, latent_dim, ch, resolution, num_res_blocks, ch_mult, attn_resolutions)
        self.codebook = Codebook(num_codebook_vectors, latent_dim, beta)
        # self.codebook = Codebook_EAMupdate(num_codebook_vectors, latent_dim, beta)
        self.quant_conv = nn.Conv2d(latent_dim, latent_dim, 1)
        self.post_quant_conv = nn.Conv2d(latent_dim, latent_dim, 1)
        # if ckpt_path is not None:
        #     sd = torch.load(ckpt_path, map_location="cpu")
        #     self.load_state_dict(sd, strict=False)
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.GroupNorm):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)


    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        quantized_codebook_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(quantized_codebook_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, x):
        encoded_images = self.encoder(x)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, index):
        z = self.codebook.dictionary_lookup(index)
        quantized_codebook_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(quantized_codebook_mapping)
        return decoded_images

    def calculate_lambda(self, nll_loss, g_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight

        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        print("Loaded Checkpoint for VQGAN....")


