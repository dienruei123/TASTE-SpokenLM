import os
import yaml
import torch
import shutil
import logging
import argparse
from tqdm import tqdm
from functools import partial, reduce
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from transformers.integrations import TensorBoardCallback
from taslm.utils_taslm import pad_seq_collate_fn, get_lora_config, prepare_dataset, pad_seq_collate_fn_for_taste, pad_seq_collate_fn_for_taste_repeat
from taslm.modeling_taslm import TaslmForCausalLM
from taslm.configuration_taslm import TaslmConfig


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


class TaslmTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

# TODO: modify to use the call back
class CustomEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, tb_writer, test_on_begin, eval_steps):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tb_writer = tb_writer
        self.test_on_begin = test_on_begin
        self.eval_steps = eval_steps
        self.eval_best_score = 0.0
        self.eval_best_speech_loss = float('inf')
        self.eval_best_total_loss = float('inf')
        self.eval_best_text_loss = float('inf')
        self.exp_dir = trainer.args.output_dir

    def on_step_begin(self, args, state, control, **kwargs):
        if self.test_on_begin and state.global_step == 0:
            self._evaluate(state)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == self.eval_steps - 1:
            self._evaluate(state)

    def _evaluate(self, state):
        model = self.trainer.model
        eval_dataloader = self.trainer.get_eval_dataloader(self.eval_dataset)
        accelerator = self.trainer.accelerator

        model.eval()
        with torch.inference_mode():
            text_corrects = torch.tensor(0.0, device=model.device)
            text_totals = torch.tensor(0.0, device=model.device)
            speech_corrects_list, speech_totals_list = [], []
            loss_dict_list = []
            
            for i in range(model.speech_num_channels):
                speech_corrects = torch.tensor(0.0, device=model.device)
                speech_totals = torch.tensor(0.0, device=model.device)
                speech_corrects_list.append(speech_corrects)
                speech_totals_list.append(speech_totals)
            
            for batch in (tqdm(eval_dataloader, desc='evaluation')
                          if accelerator.is_main_process else eval_dataloader):
                outputs = model(**batch)
                loss_dict_list.append(outputs.get('loss_dict', {}))

                text_preds = outputs['text_logits'][..., :-1, :].argmax(-1).contiguous().view(-1)
                text_labels = batch['text_labels'][..., 1:].contiguous().view(-1)
                assert len(text_preds) == len(text_labels)
                text_mask = (text_labels != model.ignore_index)

                speech_labels_mask = batch.get('speech_labels_mask', None)
                if speech_labels_mask is not None:
                    speech_labels_mask = speech_labels_mask[..., 1:].contiguous().view(-1)

                if model.speech_token_adopt_latent_sampling:
                    speech_y_pred = outputs['speech_y_pred'][..., :-1, :].contiguous().view(-1, model.latent_dim)
                    _quantized, speech_indices, _loss = model.speech_embed_tokens.rvq(speech_y_pred)
                    for i in range(model.speech_num_channels):
                        speech_labels = batch['speech_labels'][..., 1:, i].contiguous().view(-1)
                        speech_mask = (speech_labels != model.ignore_index)
                        if speech_labels_mask is not None:
                            speech_mask = torch.logical_and(speech_mask, speech_labels_mask)
                        # calculate accuracy of each channel
                        _cur_speech_indices = speech_indices[..., i]
                        _cur_speech_scores = (_cur_speech_indices == speech_labels)[speech_mask].float()
                        speech_corrects_list[i] += _cur_speech_scores.sum()
                        speech_totals_list[i] += _cur_speech_scores.size(0)
                else:
                    speech_logits = outputs['speech_logits']
                    for i in range(model.speech_num_channels):
                        _channel_start_idx, _channel_end_idx = model.config.speech_vocab_size * i, model.config.speech_vocab_size * (i+1)
                        _cur_speech_logits = speech_logits[..., :, _channel_start_idx:_channel_end_idx]
                        speech_preds = _cur_speech_logits[..., :-1, :].argmax(-1).contiguous().view(-1)
                        if model.speech_num_channels > 1:
                            speech_labels = batch['speech_labels'][..., 1:, i].contiguous().view(-1)
                        else:
                            speech_labels = batch['speech_labels'][..., 1:].contiguous().view(-1)

                        speech_mask = (speech_labels != model.ignore_index)
                        if speech_labels_mask is not None:
                            speech_mask = torch.logical_and(speech_mask, speech_labels_mask)

                        _cur_speech_scores = (speech_preds == speech_labels)[speech_mask].float()
                        speech_corrects_list[i] += _cur_speech_scores.sum()
                        speech_totals_list[i] += _cur_speech_scores.size(0)
                
                text_scores = (text_preds == text_labels)[text_mask].float()
                text_corrects += text_scores.sum()
                text_totals += text_scores.size(0)

        gathered_text_correct = accelerator.gather(text_corrects)
        gathered_text_total = accelerator.gather(text_totals)
        for i in range(model.speech_num_channels):
            speech_corrects_list[i] = accelerator.gather(speech_corrects_list[i])
            speech_totals_list[i] = accelerator.gather(speech_totals_list[i])
            # gathered_speech_correct = accelerator.gather(speech_corrects)
            # gathered_speech_total = accelerator.gather(speech_totals)

        if accelerator.is_main_process:
            text_acc = gathered_text_correct.sum().cpu().item() / gathered_text_total.sum().cpu().item()
            # speech_acc = gathered_speech_correct.sum().cpu().item() / gathered_speech_total.sum().cpu().item()
            entry = {
                'epoch': state.epoch,
                'eval_text_token_accuracy': text_acc,
                # 'eval_speech_token_accuracy': speech_acc,
                'text_token_lengths': gathered_text_total.sum().cpu().item(),
                # 'speech_token_lengths': gathered_speech_total.sum().cpu().item(),
            }
            for i in range(model.speech_num_channels):
                gathered_speech_correct, gathered_speech_total = speech_corrects_list[i], speech_totals_list[i]
                _cur_speech_acc = gathered_speech_correct.sum().cpu().item() / gathered_speech_total.sum().cpu().item()
                entry[f'eval_speech_token_accuracy.{i}'] = _cur_speech_acc
                if i == 0:
                    entry['speech_token_lengths'] = gathered_speech_total.sum().cpu().item(),
            # gather loss dict
            len_loss_dicts = len(loss_dict_list)
            def _merge_loss_dicts(d1, d2):
                keys = set(d1) | set(d2)
                return {k: (d1.get(k, 0.0) + d2.get(k, 0.0)) for k in keys}
            # merge_loss_dicts_fn = partial(_merge_loss_dicts, total_length=len_loss_dicts)
            reduced_loss_dict = reduce(_merge_loss_dicts, loss_dict_list) # reduced by sum
            for k, v in reduced_loss_dict.items():
                reduced_loss_dict[k] = v / len_loss_dicts # calculate avergage
            state.log_history.append(entry)
            self.tb_writer.add_scalar(
                'eval/text_token_accuracy', 
                text_acc, 
                state.global_step
            )
            for i in range(model.speech_num_channels):
                _speech_acc = entry[f'eval_speech_token_accuracy.{i}']
                self.tb_writer.add_scalar(
                    f'eval/speech_token_accuracy.{i}', 
                    _speech_acc, 
                    state.global_step
                )
                if i == 0:
                    speech_acc = _speech_acc
            text_loss, speech_loss, total_loss = 0.0, 0.0, 0.0
            for key, val in reduced_loss_dict.items():
                self.tb_writer.add_scalar(
                    f'eval/{key}',
                    val,
                    state.global_step
                )
                if 'speech' in key:
                    speech_loss += val
                if 'text' in key:
                    text_loss += val
                total_loss += val

            print(f"Test: {entry}")
            print(f"Loss: {reduced_loss_dict}")
            if speech_acc > self.eval_best_score:
                _save_best_dir = os.path.join(self.exp_dir, "checkpoint-best")
                self.trainer.save_model(_save_best_dir, _internal_call=True)
                self.eval_best_score = speech_acc
                print(f"Save new checkpoint best speech acc.")
            if speech_loss < self.eval_best_speech_loss:
                _save_best_dir = os.path.join(self.exp_dir, "checkpoint-best-speech-loss")
                self.trainer.save_model(_save_best_dir, _internal_call=True)
                self.eval_best_speech_loss = speech_loss
                print(f"Save new checkpoint best speech loss.")
            if text_loss < self.eval_best_text_loss:
                _save_best_dir = os.path.join(self.exp_dir, "checkpoint-best-text-loss")
                self.trainer.save_model(_save_best_dir, _internal_call=True)
                self.eval_best_text_loss = text_loss
                print(f"Save new checkpoint best text loss.")
            if total_loss < self.eval_best_total_loss:
                _save_best_dir = os.path.join(self.exp_dir, "checkpoint-best-total-loss")
                self.trainer.save_model(_save_best_dir, _internal_call=True)
                self.eval_best_total_loss = total_loss
                print(f"Save new checkpoint best total loss.")
        model.train()
        accelerator.wait_for_everyone()


def main(training_config):
    # prepare config and model
    slm_config_dict = training_config.slm_config_dict
    taslm_config = TaslmConfig(**slm_config_dict)
    attn_implementation = taslm_config.attn_implementation
    torch_dtype = eval(f"torch.{taslm_config.torch_dtype}")
    logging.info(f"attn_implementation: {attn_implementation}")
    logging.info(f"torch_dtype: {torch_dtype}")
    # prepare exp dir and tb dir
    exp_root = training_config.exp_root
    exp_name = training_config.exp_name
    exp_dir = os.path.join(exp_root, exp_name)
    setattr(training_config, 'exp_dir', exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy(training_config.config, exp_dir)
    tb_root = training_config.tensorboard_root
    tb_dir = os.path.join(tb_root, exp_name)
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_dir) # create tb writer
    # save config
    taslm_config.save_pretrained(exp_dir)
    # prepare training args
    # assert training_config.actual_batch_size == \
    #     (training_config.micro_batch_size * training_config.get("gradient_accumulation_steps", 1) * torch.cuda.device_count())
    # steps_per_epoch = len(train_dataset) // training_config.actual_batch_size
    # eval_steps = training_config.eval_steps if training_config.eval_steps else int(training_config.eval_epochs * steps_per_epoch)
    eval_steps = training_config.eval_steps
    training_args = TrainingArguments(
        output_dir=exp_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=float(training_config.learning_rate),
        lr_scheduler_type=training_config.lr_scheduler,
        warmup_steps=training_config.warmup_steps,
        weight_decay=training_config.weight_decay,
        max_grad_norm=training_config.max_grad_norm,
        num_train_epochs=training_config.num_epochs,
        logging_strategy='steps',
        logging_steps=training_config.logging_steps,
        logging_dir=tb_dir,
        eval_strategy='no',
        save_strategy='steps',
        save_steps=eval_steps,
        per_device_eval_batch_size=training_config.test_micro_batch_size,
        per_device_train_batch_size=training_config.micro_batch_size,
        seed=training_config.seed,
        no_cuda=False,
        optim=training_config.optimizer,
        deepspeed=training_config.deepspeed,
        report_to='none',
        bf16=training_config.use_bf16,
        bf16_full_eval=training_config.use_bf16,
        gradient_checkpointing=training_config.gradient_checkpointing,
        use_liger_kernel=training_config.use_liger_kernel,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        remove_unused_columns=False,
        # batch_eval_metrics=True,
    )
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(taslm_config.llama_pretrained_dir)
    # init model
    taslm_model = TaslmForCausalLM._from_config(taslm_config, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    logging.info(taslm_model)
    # to peft model
    _lora_config = get_lora_config(taslm_model.language_model, training_config)
    taslm_model.apply_lora(_lora_config, training_config)
    logging.info(taslm_model)
    if getattr(training_config, "load_from_pretrained_dir", None) is not None:
        logging.info(f"loading pretrained taslm from dir={training_config.load_from_pretrained_dir}")
        from safetensors.torch import load_file
        ckpt_fpath = os.path.join(training_config.load_from_pretrained_dir, "model.safetensors")
        _state_dict = load_file(ckpt_fpath)
        taslm_model.load_state_dict(_state_dict)

    # prepare datasets
    train_dataset = prepare_dataset(training_config.train_data_list)
    eval_dataset = prepare_dataset(training_config.eval_data_list)
    # prepare dataloader
    collate_fn_kwargs = training_config.collate_fn_kwargs
    collate_fn_name = getattr(training_config, "collate_fn_name", "pad_seq_collate_fn")
    collate_fn = eval(collate_fn_name)
    logging.info(f"collate_fn_name: {collate_fn_name}")
    logging.info(f"collate_fn_kwargs: {collate_fn_kwargs}")
    partial_fn_for_collate = partial(
        collate_fn,
        **collate_fn_kwargs,
    )

    trainer = TaslmTrainer(
        model=taslm_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=partial_fn_for_collate,
    )
    # add eval and tb callback
    trainer.add_callback(CustomEvalCallback(trainer, eval_dataset, tb_writer, training_config.test_on_begin, eval_steps))
    trainer.add_callback(TensorBoardCallback(tb_writer))
    # start training
    resume_from_checkpoint = getattr(training_config, 'resume_dir', None)
    if resume_from_checkpoint is not None:
        print(f"will resume from {resume_from_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/train_config.yml', type=str)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'rb').read())
    for key, val in config.items():
        if not hasattr(args, key):
            setattr(args, key, val)

    main(args)
