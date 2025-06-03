import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseFusion(nn.Module):
    def __init__(
        self,
        hidden_size=None,
        normalize_speech=False, # considered that the speech features is more like high dimensional features, we may need to normalize the speech embeddings
        dropout_speech=0.0,
    ):
        super().__init__()
        if normalize_speech:
            self.layer_norm_speech = nn.LayerNorm(hidden_size)
        self.conduct_dropout_speech = dropout_speech > 0.0
        if self.conduct_dropout_speech:
            self.dropout_speech = nn.Dropout1d(p=dropout_speech)
        self.normalize_speech = normalize_speech
        print(f"[Fusion]: normalize_speech = {self.normalize_speech}")
        print(f"[Fusion]: dropout_speech = {self.conduct_dropout_speech}, rate={dropout_speech}")
    
    def forward(
        self,
        text_embeds,
        speech_embeds,
    ):
        if self.normalize_speech:
            speech_embeds = self.layer_norm_speech(speech_embeds)
        if self.conduct_dropout_speech:
            speech_embeds = self.dropout_speech(speech_embeds)
        combined_embeds = text_embeds + speech_embeds
        return combined_embeds


class GatedFusion(BaseFusion):
    def __init__(
        self,
        hidden_size=None,
        text_residual=False,
        **kwargs_for_basic_fusion
    ):
        super().__init__(hidden_size=hidden_size, **kwargs_for_basic_fusion)
        self.gate = nn.Linear(2 * hidden_size, 2, bias=False)
        self.text_residual = text_residual

    
    def forward(
        self,
        text_embeds,
        speech_embeds,
    ):
        if self.normalize_speech:
            speech_embeds = self.layer_norm_speech(speech_embeds)
        if self.conduct_dropout_speech:
            speech_embeds = self.dropout_speech(speech_embeds)
        gate = torch.sigmoid(self.gate(torch.cat([text_embeds, speech_embeds], dim=-1)))
        combined_embeds = gate[..., 0:1] * text_embeds + gate[..., 1:2] * speech_embeds
        if self.text_residual:
            combined_embeds = combined_embeds + text_embeds
        return combined_embeds


class WeightedSumFusion(BaseFusion):
    def __init__(
        self,
        hidden_size=None,
        weight_init_type: str = 'zero_audio',
        **kwargs_for_basic_fusion
    ):
        super().__init__(hidden_size=hidden_size, **kwargs_for_basic_fusion)
        if weight_init_type == 'balance':
            self.weights = nn.Parameter(torch.tensor([1., 1.]), requires_grad=True)
        elif weight_init_type == 'zero_audio':
            self.weights = nn.Parameter(torch.tensor([-2., 2.]), requires_grad=True)
    
    def forward(
        self,
        text_embeds,  # [b, s, 3072]
        audio_embeds  # [b, s, 512]
    ):
        weights = F.softmax(self.weights, dim=0)
        weights = weights.view(2, 1, 1, 1)

        inputs = torch.stack([audio_embeds, text_embeds], dim=0)
        fused = (weights * inputs).sum(dim=0)
        return fused


FUSION_METHOD_CLASS_MAP = {
    'addition': BaseFusion,
    'gated': GatedFusion,
    'weighted_sum': WeightedSumFusion,
}


class LatentSamplingLayer(nn.Module): # reference from MELLE
    def __init__(
        self, 
        lm_hidden_dim=None,
        latent_dim=256,
        fc_mu_requires_bias=True,
        b_logvar_is_linear=False,
        use_additional_mlp_and_residual=False,
        conduct_reparameterization=True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(lm_hidden_dim, latent_dim, bias=fc_mu_requires_bias)
        if b_logvar_is_linear:
            self.b_logvar = nn.Linear(lm_hidden_dim, latent_dim, bias=False)
        else:
            self.b_logvar = nn.Parameter(torch.zeros(latent_dim))
        self.b_logvar_is_linear = b_logvar_is_linear
        self.fc_mu_requires_bias = fc_mu_requires_bias
        self.use_additional_mlp_and_residual = use_additional_mlp_and_residual
        self.conduct_reparameterization = conduct_reparameterization
        if self.use_additional_mlp_and_residual:
            self.mlp = nn.Sequential(
                [
                    nn.Linear(latent_dim)
                ]
            )
    

    def reparameterize(self, mu, sigma):
        orig_value = mu + sigma
        epsilon = torch.randn_like(sigma)
        _sampled_value = mu + sigma * epsilon
        new_value = orig_value + (_sampled_value - orig_value).detach()
        return new_value
    
    
    def forward(self, hidden_states):
        bsz, tsz, csz = hidden_states.shape
        # compute the mean
        mu = self.fc_mu(hidden_states)
        # get the logvar
        if self.b_logvar_is_linear:
            logvar = self.b_logvar(hidden_states)
            sigma = torch.exp(0.5 * logvar)
        else:
            logvar = self.b_logvar
            sigma = torch.exp(0.5 * logvar).view(1, 1, -1).expand_as(mu)
        # reparameterize to get the sampled latent z
        if self.conduct_reparameterization:
            z = self.reparameterize(mu, sigma)
        else:
            z = mu + sigma
        # y_pred = self.mlp(z) + z # currently we don't have additional mlp for residual 
        return mu, logvar, z

        # total_loss = l_reg + l_kl
        # losses = {
        #     'l_reg': l_reg,
        #     'l_kl': l_kl,
        #     'total_loss': total_loss,
        # }
        
    

