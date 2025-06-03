import os
from transformers import logging, LlamaConfig, PretrainedConfig

logger = logging.get_logger(__name__)

class TaslmConfig(PretrainedConfig):

    def __init__(
        self, 
        llama_pretrained_dir="/proj/mtklmadm/models/Llama-3.2-3B",
        llama_use_liger_kernel=False,
        # taslm specific kwargs
        speech_token_size=4096, # without padding
        speech_vocab_size=None,
        speech_num_channels=1,
        fusion_method='addition',
        fusion_kwargs={}, 
        speech_token_type='s3',
        speech_tokenizer_pretrained_dir=None, # if set, initialize the embedding from the tokenizer dir (rvq layers)
        speech_tokenizer_hidden_size=None, # set this properly to allow init speech token embeds from the rvq layers
        speech_embed_directly_use_rvq=False,
        speech_labels_apply_quantization=False,
        speech_token_adopt_latent_sampling=False,
        speech_latent_sampler_kwargs=None,
        speech_embed_dropout=0.0,
        speech_loss_apply_mask=True,
        speech_tokenizer_rvq_kwargs=None,
        speech_multi_channel_loss_decay_factor=None,
        text_conduct_kl_loss=False,
        attn_implementation='flash_attention_2',
        **kwargs,
    ):
        super().__init__()
        self.llama_pretrained_dir = llama_pretrained_dir
        self.llama_config = LlamaConfig.from_pretrained(self.llama_pretrained_dir)
        self.llama_use_liger_kernel = llama_use_liger_kernel
        self.speech_token_size = speech_token_size
        self.speech_vocab_size = speech_vocab_size
        self.speech_num_channels = speech_num_channels
        self.fusion_method = fusion_method
        self.fusion_kwargs = fusion_kwargs
        self.speech_token_type = speech_token_type
        self.speech_tokenizer_pretrained_dir = speech_tokenizer_pretrained_dir
        self.speech_tokenizer_hidden_size = speech_tokenizer_hidden_size
        self.speech_embed_directly_use_rvq = speech_embed_directly_use_rvq
        self.speech_labels_apply_quantization = speech_labels_apply_quantization
        self.speech_token_adopt_latent_sampling = speech_token_adopt_latent_sampling
        self.speech_latent_sampler_kwargs = speech_latent_sampler_kwargs
        self.speech_loss_apply_mask = speech_loss_apply_mask
        self.speech_tokenizer_rvq_kwargs = speech_tokenizer_rvq_kwargs
        self.speech_multi_channel_loss_decay_factor = speech_multi_channel_loss_decay_factor
        self.text_conduct_kl_loss = text_conduct_kl_loss
        self.attn_implementation = attn_implementation
        # kwargs dependent props
        self.hidden_size = self.llama_config.hidden_size
        self.torch_dtype = self.llama_config.torch_dtype
        self.text_vocab_size = self.llama_config.vocab_size
        if self.fusion_kwargs.get('hidden_size', None) is None:
            self.fusion_kwargs['hidden_size'] = self.hidden_size
        if self.speech_vocab_size is None:
            self.speech_vocab_size = self.speech_token_size + 2 # bos and eos
        if self.speech_tokenizer_rvq_kwargs is not None: # ensure consistency between rvq out and embed proj
            assert self.speech_tokenizer_rvq_kwargs['dim'] == self.speech_tokenizer_hidden_size, f"hidden size mismatch between rvq_kwargs.dim ({self.speech_tokenizer_rvq_kwargs.dim}) and speech_tokenizer_hidden_size ({self.speech_tokenizer_hidden_size})."
        self.speech_bos_token_id = self.speech_token_size
        self.speech_eos_token_id = self.speech_token_size + 1
    

if __name__ == "__main__":
    pretrained_llama_dir = "/proj/mtklmadm/models/Llama-3.2-3B"
    config = TaslmConfig.from_pretrained(pretrained_llama_dir)
    slm_config_dict = {
        'speech_token_size': 4096,
        'speech_num_channels': 1,
        'fusion_method': 'addition',
        'speech_token_type': 's3',
    }
    for key, val in slm_config_dict.items():
        setattr(config, key, val)
    from pprint import pp
    pp(config)
    tmp_dir = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/SpokenLM/conf/baseline_s3_padding"
    os.makedirs(tmp_dir, exist_ok=True)
    config.save_pretrained(tmp_dir)
    new_config = TaslmConfig.from_pretrained(tmp_dir)
    pp(new_config)
    aligned = True
    _new_config_dict = new_config.to_dict()
    for key, val in config.to_dict().items():
        if val != _new_config_dict[key]:
            aligned = False
            break
    pp(f"Config is aligned before and after saving: {aligned}")



