#### TL;DR: 
#### In stage2 of taste, we aim to investigate the possibility of different quantization strategies, and the model will be initialized from the continuous one (e.g., taste_wrap-skip_word_no-vq_concat).

---

## Exps
### RVQ
#### RVQ-CODEBOOK_DIM-LAYER_NUM-CODEBOOK_SIZE
1. Study of codebook dim (full finetune)
* RVQ-ddefault-l4-k256
* RVQ-d512-l4-k256
* RVQ-d256-l4-k256
2. study of full-finetune or partial finetune
* RVQ-default-l4-k256 full
* RVQ-default-l4-k256 partial (text_encoder + quantizer + llm only)
* RVQ-default-l4-k256 partial (quantizer + llm only)