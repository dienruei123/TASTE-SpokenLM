import os
import re
import torch
from pprint import pp
from transformers import LlamaForCausalLM, AutoTokenizer
from taslm.modeling_taslm import TaslmForCausalLM
from taslm.configuration_taslm import TaslmConfig
from taslm.utils_taslm import fake_baseline_data_generator, load_training_config, get_lora_config


def test_toggle_adapter_layers(taslm_model, llama_model, llm_tokenizer, device='cuda'):
    prompt = "Hey, are you conscious? Can you talk to me?"
    text_prompt_input_ids = llm_tokenizer.encode(prompt)
    text_prompt_input_ids = torch.tensor(text_prompt_input_ids).to(device).view(1, -1)
    print(f"text_prompt_input_ids: {text_prompt_input_ids.shape}")
    # test manually disable
    # taslm_model.eval()
    # llama_model.eval()
    with torch.inference_mode():
        embed_tokens = taslm_model.language_model.get_input_embeddings()
        text_embeds_for_taslm = embed_tokens(text_prompt_input_ids)
        taslm_model.language_model.base_model.disable_adapter_layers()
        _decoder = taslm_model.language_model.base_model.get_decoder()
        # _decoder = taslm_model.language_model.get_decoder()
        orig_text_only_hidden_states = _decoder(
            inputs_embeds = text_embeds_for_taslm,
            # attention_mask = None,
            # past_key_values = None,
            # use_cache = None, 
            # **kwargs,
        )[0] # take out the hidden states
        orig_text_only_logits = taslm_model.language_model.base_model.lm_head(orig_text_only_hidden_states)
        # orig_text_only_logits = taslm_model.language_model.lm_head(orig_text_only_hidden_states)
        # shift_orig_text_only_log_prob = F.log_softmax(orig_text_only_logits[..., :-1, :80000].contiguous(), dim=-1).detach()
        taslm_model.language_model.base_model.enable_adapter_layers()
        # llama 2
        llama_embed_tokens = llama_model.get_input_embeddings()
        text_embeds_for_llama = llama_embed_tokens(text_prompt_input_ids)
        # check embeddings' consistency
        llama_decoder = llama_model.get_decoder()
        llama_text_hidden_states = llama_decoder(
            inputs_embeds = text_embeds_for_llama,
        )[0]
        llama_text_logits = llama_model.lm_head(llama_text_hidden_states)
        # embeds
        pp(text_embeds_for_taslm)
        pp(text_embeds_for_taslm.shape)
        pp(text_embeds_for_llama)
        pp(text_embeds_for_llama.shape)
        print((text_embeds_for_taslm == text_embeds_for_llama).sum())
        # hidden states
        pp(orig_text_only_hidden_states)
        pp(orig_text_only_hidden_states.shape)
        pp(llama_text_hidden_states)
        pp(llama_text_hidden_states.shape)
        print((orig_text_only_hidden_states == llama_text_hidden_states).sum())
        print("--------------------------------")
        for taslm_hidden, llama_hidden in zip(orig_text_only_hidden_states[0], llama_text_hidden_states[0]):
            pp(taslm_hidden)
            pp(llama_hidden)
            print("--------------------------------")
        # test generate
        for i in range(5):
            taslm_generate_ids = taslm_model.language_model.base_model.generate(text_prompt_input_ids, max_length=30)
            llama_generate_ids = llama_model.generate(text_prompt_input_ids, max_length=30)
            print("test generation")
            print(llm_tokenizer.batch_decode(taslm_generate_ids)[0])
            print("--------------------")
            print(llm_tokenizer.batch_decode(llama_generate_ids)[0])
        # logits
        # pp(orig_text_only_logits)
        # pp(orig_text_only_logits.shape)
        # pp(llama_text_logits)
        # pp(llama_text_logits.shape)
        # print((orig_text_only_logits == llama_text_logits).sum())

def main():
    pretrained_dir = "./taslm/exp/0303_taslm_1B_eos-rvq_word-delay_gated-fusion_drop-proj_text-kl_latent_r64"
    llama_pretrained_dir = "/proj/mtklmadm/models/mtk53678/Llama-3.2-1B"
    # investigate llama tokenizer
    # llm_tokenizer = AutoTokenizer.from_pretrained(llama_pretrained_dir)
    # def is_valid_word(word):
    #     pattern = r"^[A-Za-z0-9,\.!?'\- ]+$"
    #     return bool(re.fullmatch(pattern, word))
    # for i in range(128000, 0, -1):
    #     subword = llm_tokenizer.decode(i)
    #     print(subword)
    #     if is_valid_word(subword):
    #         print(i)
    #         print(subword)
    #         print(llm_tokenizer.decode(i+1))
    #         break
    # return
    # settings
    torch_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'
    device = f"cuda" if torch.cuda.is_available() else 'cpu'
    # prepare model
    ## load config
    taslm_config = TaslmConfig.from_pretrained(pretrained_dir)
    ## load model from config
    taslm_model = TaslmForCausalLM._from_config(taslm_config)
    ## load training config -> get lora config -> apply lora
    training_config = load_training_config(pretrained_dir)
    _lora_config = get_lora_config(taslm_model.language_model, training_config)
    taslm_model.apply_lora(_lora_config, training_config)
    ## load from state dict
    state_dict_fpath = os.path.join(pretrained_dir, 'checkpoint-best/model.safetensors')
    from safetensors.torch import load_file
    _state_dict = load_file(state_dict_fpath)
    taslm_model.load_state_dict(_state_dict)
    ## register speech tokenizer and asr model
    taslm_model = taslm_model.to(device)
    # pp(taslm_model)
    # pp(taslm_model.language_model.base_model.model.model.embed_tokens)
    # pp(taslm_model.language_model.base_model.model.lm_head.base_layer)
    # for name, param in taslm_model.language_model.base_model.model.model.embed_tokens.named_parameters():
    #     print(name, param)
    # prepare data
    # fake_data_for_baseline = fake_baseline_data_generator(
    #     taslm_model.config.eos_token_id,
    #     taslm_model.config.speech_eos_token_id,
    #     ensure_alignment=False
    # )
    # pp(fake_data_for_baseline)
    # fake_data_aligned_for_baseline = fake_baseline_data_generator(
    #     taslm_model.config.eos_token_id,
    #     taslm_model.config.speech_eos_token_id,
    #     ensure_alignment=True,
    # )
    # pp(fake_data_aligned_for_baseline)
    # load llama for comparison
    llama_model = LlamaForCausalLM.from_pretrained(llama_pretrained_dir, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    llama_model.to(device)
    # pp(llama_model)
    # pp(llama_model.model.embed_tokens)
    # pp(llama_model.lm_head)
    # check embedding
    # for name, param in llama_model.model.embed_tokens.named_parameters():
    #     print(name, param)
    # for name, param in taslm_model.language_model.base_model.model.model.embed_tokens.named_parameters():
    #     print(name, param)
    # # check linear head
    # for name, param in llama_model.lm_head.named_parameters():
    #     print(name, param)
    # for name, param in taslm_model.language_model.base_model.model.lm_head.base_layer.named_parameters():
    #     print(name, param)
    # for name, param in taslm_model.speech_embed_tokens.named_parameters():
    #     print(name, torch.mean(param), torch.norm(param), torch.max(param), torch.min(param))
    # test disable adapter layer
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_pretrained_dir)
    test_toggle_adapter_layers(taslm_model, llama_model, llama_tokenizer, device=device)
    return
    # _llama_model = LlamaForCausalLM.from_pretrained(_llama_pretrained_dir, torch_dtype=torch.float16)
    # to eval mode
    taslm_model.eval()
    llama_model.eval()
    _llama_model.eval()
    # to different devices
    print("sending models to cuda...")
    taslm_device = "cuda:0"
    llama_device = "cuda:1"
    _llama_device = "cuda:2"
    taslm_model.to(taslm_device)
    llama_model.to(llama_device)
    _llama_model.to(_llama_device)
    print("Done.")
    # test generation
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = llama_tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids
    generate_ids = llama_model.generate(input_ids.to(llama_device), max_length=30)
    _generate_ids = _llama_model.generate(input_ids.to(_llama_device), max_length=30)
    print("test generation")
    print(llama_tokenizer.batch_decode(generate_ids)[0])
    print("--------------------")
    print(llama_tokenizer.batch_decode(_generate_ids)[0])
    # test forward results
    with torch.inference_mode():
        print("=================== text-only ====================")
        data = fake_data_for_baseline
        llama_result = llama_model(
            input_ids = data['text_input_ids'].to(llama_device),
            attention_mask = data['text_attention_mask'].to(llama_device),
        )
        pp(llama_result.logits)
        print("--------------------------------------------------")
        _llama_result = _llama_model(
            input_ids = data['text_input_ids'].to(_llama_device),
            attention_mask = data['text_attention_mask'].to(_llama_device),
        )
        pp(_llama_result.logits)
        print("--------------------------------------------------")
        taslm_result = taslm_model(
            text_input_ids = data['text_input_ids'].to(taslm_device),
            text_attention_mask = data['text_attention_mask'].to(taslm_device),
        )
        pp(taslm_result['text_logits'])
        print("=================== with speech ===================")
        taslm_result_with_speech_tokens = taslm_model(
            text_input_ids = data['text_input_ids'].to(taslm_device),
            text_attention_mask = data['text_attention_mask'].to(taslm_device),
            speech_input_ids = data['speech_input_ids'].to(taslm_device),
            speech_attention_mask = data['speech_attention_mask'].to(taslm_device),
        )
        pp(taslm_result_with_speech_tokens['text_logits'])
        pp(taslm_result_with_speech_tokens['speech_logits'])
    
    




if __name__ == "__main__":
    main()