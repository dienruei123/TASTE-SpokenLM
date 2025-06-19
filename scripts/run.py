# coding: utf-8

import argparse
import shutil
from pathlib import Path
import os
import re
from pprint import pprint
import glob
import json

from tqdm import tqdm
import yaml
from datasets import load_from_disk, concatenate_datasets, Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.integrations import TensorBoardCallback
from transformers import WhisperProcessor, AutoTokenizer
import torch.nn.functional as F

from taste_speech import TasteConfig, TasteSpokenLMConfig, TasteForCausalLM, TasteSpokenLM, TasteProcessor
from taste_speech.modules_taste.cosyvoice.utils import IGNORE_ID  # -1
from taste_speech.data.dataset import TasteDataset


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

LLAMA_PATH = os.path.abspath('./STAGE1_TRAIN/storage/pretrained_models/Llama-3.2-1B/')
WHISPER_PATH = os.path.abspath('./STAGE1_TRAIN/storage/pretrained_models/whisper-large-v3')
DISTIL_WHISPER_PATH = os.path.abspath('./STAGE1_TRAIN/storage/pretrained_models/distil-large-v3')


class TasteTrainer(Trainer):
    def __init__(self, data_stage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_stage = data_stage
        self.ref_model = None

    def set_ref_model(self, ref_model):
        self.ref_model = ref_model.to(LOCAL_RANK)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.ref_model is not None:
            outputs = model(**inputs, ref_model=self.ref_model)
        else:
            outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def _accum_metrics_acc(self, logits, labels, metric_dict):
        predictions = logits.argmax(-1)
        mask = (labels != IGNORE_ID)
        scores = (predictions == labels)[mask].float()
        metric_dict['corrects'] += scores.sum()
        metric_dict['totals'] += scores.size(0)

    def _accum_metrics_mse(self, taste_logits, taste_labels, metric_dict):
        # taste_logits: B, T, layer, k
        # taste_labels: B, T, layer
        predictions = taste_logits.argmax(-1)
        mask = (taste_labels != IGNORE_ID).all(dim=-1)
        vq_module = self.model.audio_tower.vq.rvq
        distance = F.mse_loss(
            vq_module.get_output_from_indices(predictions[mask]),
            vq_module.get_output_from_indices(taste_labels[mask]),
        )
        totals = mask.sum()
        metric_dict['distance'] += distance
        metric_dict['totals'] += totals

    def _get_init_metrics(self, device):
        if self.data_stage == 1:
            metrics = {
                'speech_token_accuracy': {
                    'corrects': torch.tensor(0.0, device=device),
                    'totals': torch.tensor(0.0, device=device)
                }
            }
        else:
            metrics = {
                'text_accuracy': {
                    'corrects': torch.tensor(0.0, device=device),
                    'totals': torch.tensor(0.0, device=device)
                },
                'audio_mse': {
                    'distance': torch.tensor(0.0, device=device),
                    'totals': torch.tensor(0.0, device=device)
                },
                'a0_accuracy': {
                    'corrects': torch.tensor(0.0, device=device),
                    'totals': torch.tensor(0.0, device=device)
                },
                'a1_accuracy': {
                    'corrects': torch.tensor(0.0, device=device),
                    'totals': torch.tensor(0.0, device=device)
                },
                'a2_accuracy': {
                    'corrects': torch.tensor(0.0, device=device),
                    'totals': torch.tensor(0.0, device=device)
                },
                'a3_accuracy': {
                    'corrects': torch.tensor(0.0, device=device),
                    'totals': torch.tensor(0.0, device=device)
                },
                'speech_token_accuracy': {
                    'corrects': torch.tensor(0.0, device=device),
                    'totals': torch.tensor(0.0, device=device)
                }
            }
        return metrics

    def _customized_evaluate_step(self, model, inputs, metrics, show_case=False):
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        if self.data_stage == 1:
            self._accum_metrics_acc(outputs.speech_logits, outputs.speech_labels, metrics['speech_token_accuracy'])
        else:
            self._accum_metrics_acc(outputs.text_logits, outputs.text_labels, metrics['text_accuracy'])
            self._accum_metrics_acc(outputs.taste_logits[:,:,0,:], outputs.taste_labels[:,:,0], metrics['a0_accuracy'])
            self._accum_metrics_acc(outputs.taste_logits[:,:,1,:], outputs.taste_labels[:,:,1], metrics['a1_accuracy'])
            self._accum_metrics_acc(outputs.taste_logits[:,:,2,:], outputs.taste_labels[:,:,2], metrics['a2_accuracy'])
            self._accum_metrics_acc(outputs.taste_logits[:,:,3,:], outputs.taste_labels[:,:,3], metrics['a3_accuracy'])
            self._accum_metrics_mse(outputs.taste_logits, outputs.taste_labels, metrics['audio_mse'])
            if outputs.speech_logits is not None:
                self._accum_metrics_acc(outputs.speech_logits, outputs.speech_labels, metrics['speech_token_accuracy'])
            if show_case:
                print('[show case] taste prediction', outputs.taste_logits[0].argmax(-1))


    def customized_evaluate(self, ds):
        eval_dataloader = self.get_eval_dataloader(ds)
        model = self.model
        accelerator = self.accelerator

        metrics = self._get_init_metrics(model.device)
        if accelerator.is_main_process:
            eval_dataloader = tqdm(eval_dataloader, desc='evaluation')
        for i, batch in enumerate(eval_dataloader):
            self._customized_evaluate_step(model, batch, metrics, show_case=(i == 0))

        gathered_metrics = {}
        for key, d in metrics.items():
            gathered_metrics[key] = {
                k: accelerator.gather(d[k]) for k in d.keys()
            }

        for key, d in gathered_metrics.items():
            total = d['totals'].sum().cpu().item()
            if 'corrects' in d:
                correct = d['corrects'].sum().cpu().item()
                d['acc'] = correct / total if total > 0 else -1
            elif 'distance' in d:
                distance =  d['distance'].sum().cpu().item()
                d['mse'] = distance / total if total > 0 else -1

        return gathered_metrics
    

class TasteEvalTrainer(TasteTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = self._get_init_metrics(self.model.device)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):

        self._customized_evaluate_step(model, inputs, self.metrics)

        losses = None
        logits = None
        labels = None
        return losses, logits, labels


class CustomEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, tb_writer, test_on_begin, eval_steps):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tb_writer = tb_writer
        self.test_on_begin = test_on_begin
        self.eval_steps = eval_steps

    def on_step_begin(self, args, state, control, **kwargs):
        if self.test_on_begin and state.global_step == 0:
            self._evaluate(state)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == self.eval_steps - 1:
            self._evaluate(state)

    def _evaluate(self, state):
        gathered_metrics = self.trainer.customized_evaluate(self.eval_dataset)

        if self.trainer.accelerator.is_main_process:
            entry = {
                'epoch': state.epoch,
            }
            for key, d in gathered_metrics.items():
                if 'acc' in d:
                    entry[f'eval_{key}'] = d['acc']
                    self.tb_writer.add_scalar(
                        f'eval/{key}', 
                        d['acc'], 
                        state.global_step
                    )
                elif 'mse' in d:
                    entry[f'eval_{key}'] = d['mse']
                    self.tb_writer.add_scalar(
                        f'eval/{key}', 
                        d['mse'], 
                        state.global_step
                    )

            state.log_history.append(entry)
            print(f"Test: {entry}")


def pad_seq_collate_fn(batch, device=None):
    padded = {}
    for key in batch[0].keys():
        packed_list = [
            x[key][0].clone().detach() if isinstance(x[key][0], torch.Tensor) else torch.tensor(x[key][0]) 
            for x in batch
        ]
        if 'length' in key:
            padded_tensor = torch.tensor(packed_list)
        else:
            padded_tensor = pad_sequence(packed_list, batch_first=True, padding_value=0)

        padded[key] = padded_tensor.to(device) if device is not None else padded_tensor
    return padded


def _find_all_linear_names(model):
    cls = (torch.nn.Linear, )
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    output_embedding = "lm_head"
    if output_embedding in lora_module_names:  # needed for 16-bit
        lora_module_names.remove(output_embedding)

    return list(lora_module_names)


def get_lora_config(model, args, inference=False):
    from peft import LoraConfig

    lora_target_modules = list(args.lora_target_modules or [])

    if args.lora_target_linear:
        linear_names = _find_all_linear_names(model)
        print(f"found linear modules: {repr(linear_names)}")
        lora_target_modules = list(set(lora_target_modules + linear_names))

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        layers_to_transform=None,
        lora_dropout=args.lora_dropout,
        fan_in_fan_out=args.lora_fan_in_fan_out,
        modules_to_save=args.lora_modules_to_save if args.lora_modules_to_save else None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return lora_config


def _update_state_dict(model, safetensors_dir, allow_pattern=None):
    from safetensors.torch import load_file
    combined_state_dict = {}

    for filename in os.listdir(safetensors_dir):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(safetensors_dir, filename)
            
            # Load the SafeTensors file
            state_dict = load_file(file_path)
            
            # Update the combined state dict
            combined_state_dict.update(state_dict)

    combined_state_dict = {
        (k.replace('spoken_llm.', 'spoken_lm.') if k.startswith('spoken_llm') else k): v
        for k, v in combined_state_dict.items()
    }
    if 'spoken_lm.language_model.base_model.model.lm_head.weight' not in combined_state_dict and 'spoken_lm.language_model.base_model.model.model.embed_tokens.weight' in combined_state_dict:
        combined_state_dict['spoken_lm.language_model.base_model.model.lm_head.weight'] = combined_state_dict['spoken_lm.language_model.base_model.model.model.embed_tokens.weight']
    if 'spoken_lm.language_model.lm_head.weight' not in combined_state_dict and 'spoken_lm.language_model.model.embed_tokens.weight' in combined_state_dict:
        combined_state_dict['spoken_lm.language_model.lm_head.weight'] = combined_state_dict['spoken_lm.language_model.model.embed_tokens.weight']

    assert len(combined_state_dict) > 0

    if allow_pattern:
        new_state_dict = {
            k: combined_state_dict[k] if k.startswith(allow_pattern) else v 
            for k, v in model.state_dict().items()
        }
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(combined_state_dict)
    return model


def _copy_tie_embedding(model):
    llm = model.spoken_lm.language_model
    embedding_weight = llm.model.embed_tokens.weight.clone().detach()
    llm.lm_head.weight = torch.nn.Parameter(embedding_weight.clone())
    assert llm.lm_head.weight.data_ptr() != llm.model.embed_tokens.weight.data_ptr(), "Weights are still tied!"
    return model


def prepare_model(args, model_output_dir=None, reload_model=None):
    base_model = args.base_model if not reload_model else reload_model
    if args.model_mode == 'SpokenLLM':
        model = TasteForCausalLM.from_pretrained(args.base_model, attn_implementation=args.attn_implementation)

        setattr(
            model.config, 
            "spoken_lm_config", 
            TasteSpokenLMConfig(
                **args.set_spoken_lm
            )
        )
        setattr(model, "spoken_lm_config", model.config.spoken_lm_config)
        setattr(model, "spoken_lm",
            TasteSpokenLM(
                model.config.text_config, 
                k=model.audio_tower_config.kwargs_for_quantizer['codebook_size'],
                d=model.audio_tower_config.kwargs_for_quantizer['codebook_dim'],
                sos_id=model.spoken_lm_config.sos_id,
                loss_weights=model.spoken_lm_config.loss_weights,
                delay=model.spoken_lm_config.delay,
                delay_level=model.spoken_lm_config.delay_level,
                audio_embed_conv_mode=model.spoken_lm_config.audio_embed_conv_mode,
                in_llm_module=model.spoken_lm_config.in_llm_module,
                out_llm_module=model.spoken_lm_config.out_llm_module,
                _attn_implementation = model.spoken_lm_config._attn_implementation,
            )
        )

        if hasattr(args, 'reload_llm') and args.reload_llm:
            model.reload_language_model(args.reload_llm)
    
        model = _copy_tie_embedding(model)

    else:
        model = TasteForCausalLM.from_pretrained_stage1(
            base_model, 
            attn_implementation=args.attn_implementation,
            skip_audio_in_audio_decoder=args.skip_audio_in_audio_decoder if hasattr(args, 'skip_audio_in_audio_decoder') else False,
            skip_vq_in_audio_encoder=args.skip_vq_in_audio_encoder if hasattr(args, 'skip_vq_in_audio_encoder') else False
        )

    if args.freeze_modules:
        for name, params in model.named_parameters():
            params.requires_grad = True

        freeze_modules = list(args.freeze_modules) if args.freeze_modules else []
        if 'group:language_model' in freeze_modules:
            model.freeze_language_model()
            freeze_modules.remove('group:language_model')
        if 'group:audio_tower' in freeze_modules:
            model.freeze_audio_tower()
            freeze_modules.remove('group:audio_tower')
        if 'group:speech_decoder' in freeze_modules:
            model.freeze_speech_decoder()
            freeze_modules.remove('group:speech_decoder')

        freeze_module_regexes = freeze_modules
        for name, params in model.named_parameters():
            for regex in freeze_module_regexes:
                if re.match(regex, name):
                    params.requires_grad = False
    elif args.unfreeze_modules:
        for name, params in model.named_parameters():
            params.requires_grad = False
        unfreeze_module_regexes = args.unfreeze_modules
        for name, params in model.named_parameters():
            for regex in unfreeze_module_regexes:
                if re.match(regex, name):
                    params.requires_grad = True

    if args.model_mode == 'SpokenLLM' and args.use_lora:
        lora_config = get_lora_config(model.spoken_lm.language_model, args, inference=False)
        model.apply_lora(lora_config)

        if args.reload_ckpt:
            model = _update_state_dict(model, args.reload_ckpt)

    if LOCAL_RANK == 0 and model_output_dir is not None:
        messages = [('[O] ' if params.requires_grad else '[X] ') + name for name, params in model.named_parameters()]
        with open(model_output_dir + '/weight_grad.txt', 'w') as fw:
            fw.write('\n'.join(messages))

    return model


def prepare_stage1_datasets(args, model_config, evaluate_only=False):
    train_split_dir = f"{args.stage1_data_root}/train/"
    eval_split_dir = f"{args.stage1_data_root}/dev/"

    if not evaluate_only:
        train_dataset = TasteDataset(
            train_split_dir,
            model_config.asr_config._name_or_path,
            model_config.text_config._name_or_path,
        )
    else:
        train_dataset = None
    
    eval_dataset = TasteDataset(
        eval_split_dir, 
        model_config.asr_config._name_or_path, 
        model_config.text_config._name_or_path,
    )

    return train_dataset, eval_dataset


def prepare_stage2_datasets(args, evaluate_only=False):
    root_dir = args.stage2_data_root
    selected_cols = ['llm_indices', 'llm_token_ids', 'llm_token_lengths', 'llm_word_ids']
    dev_selected_cols = selected_cols + ['speaker_embeds', 'asr_token_ids', 'asr_token_lengths',
        'asr_word_ids', 'speech_token_ids', 'speech_token_lengths']

    if not evaluate_only:
        train_dataset = concatenate_datasets(
            [
                load_from_disk(path).select_columns(selected_cols)
                for path in glob.glob(root_dir + '/trai*/*')
            ]
        )
    else:
        train_dataset = None

    eval_dataset = concatenate_datasets(
        [
            load_from_disk(path).select_columns(dev_selected_cols)
            for path in glob.glob(root_dir + '/dev/*')
        ]
    )

    return train_dataset, eval_dataset

def train(args):
    assert args.actual_batch_size == \
        (args.micro_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count())

    os.environ['HF_DATASETS_CACHE'] = args.tmp_dir
    shutil.rmtree(args.tmp_dir, ignore_errors=True)

    tensorboard_dir = f'{args.tensorboard_root_dir}/{args.job_name}/'
    model_output_dir = f'{args.exp_root_dir}/{args.job_name}/'

    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, model_output_dir + 'training_config.yml')

    model_config = TasteConfig.from_pretrained(args.base_model)

    if args.data_stage == 1:
        train_dataset, eval_dataset = prepare_stage1_datasets(args, model_config)
    elif args.data_stage == 2:
        train_dataset, eval_dataset = prepare_stage2_datasets(args, model_config)
    else:
        raise Exception

    steps_per_epoch = len(train_dataset) // args.actual_batch_size
    eval_steps = args.eval_steps if args.eval_steps else int(args.eval_epochs * steps_per_epoch)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=float(args.learning_rate),
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_epochs,
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        logging_dir=tensorboard_dir,
        save_strategy='steps',
        save_steps=eval_steps,
        per_device_eval_batch_size=args.test_micro_batch_size,
        per_device_train_batch_size=args.micro_batch_size,
        seed=args.seed,
        no_cuda=False,
        optim=args.optimizer,
        deepspeed=args.deepspeed,
        report_to='none',
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=2,
        dataloader_prefetch_factor=4,
        remove_unused_columns=False,
        # batch_eval_metrics=True,
    )

    model = prepare_model(args, model_output_dir=model_output_dir)

    tb_writer = SummaryWriter(log_dir=tensorboard_dir)

    trainer = TasteTrainer(
        data_stage=args.data_stage,
        model=model,
        args=training_args,
        tokenizer=None,
        train_dataset=train_dataset,
        data_collator=pad_seq_collate_fn,
    )
    if args.model_mode == 'SpokenLLM' and args.ref_model:
        print('using reference model ......')
        from transformers import LlamaForCausalLM
        ref_model = LlamaForCausalLM.from_pretrained(args.ref_model)
        ref_model.eval()
        trainer.set_ref_model(ref_model)

    trainer.add_callback(CustomEvalCallback(trainer, eval_dataset, tb_writer, args.test_on_begin, eval_steps))
    trainer.add_callback(TensorBoardCallback(tb_writer))

    train_result = trainer.train()
    trainer.save_state()


def evaluate(args, eval_model):
    model = prepare_model(args, reload_model=eval_model)
    model.eval()

    model_config = model.config
    if args.data_stage == 1:
        _, eval_dataset = prepare_stage1_datasets(args, model_config, evaluate_only=True)
    elif args.data_stage == 2:
        _, eval_dataset = prepare_stage2_datasets(args, model_config, evaluate_only=True)
    else:
        raise Exception

    # Define training arguments
    training_args = TrainingArguments(
        do_train=True,
        do_eval=False,
        output_dir=eval_model,
        per_device_eval_batch_size=args.test_micro_batch_size,
        deepspeed=None,
        no_cuda=False,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_prefetch_factor=4,
        remove_unused_columns=False,
    )

    # Initialize the Trainer without the train_dataset
    trainer = TasteEvalTrainer(
        data_stage=args.data_stage,
        model=model,
        tokenizer=None,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=pad_seq_collate_fn,
    )

    # Evaluate the model
    trainer.evaluate()

    gathered_metrics = {}
    for key, d in trainer.metrics.items():
        gathered_metrics[key] = {
            k: trainer.accelerator.gather(d[k]) for k in d.keys()
        }

    for key, d in gathered_metrics.items():
        total = d['totals'].sum().cpu().item()
        if 'corrects' in d:
            correct = d['corrects'].sum().cpu().item()
            d['acc'] = correct / total if total > 0 else -1
        elif 'distance' in d:
            distance =  d['distance'].sum().cpu().item()
            d['mse'] = distance / total if total > 0 else -1

    if LOCAL_RANK == 0:
        recorded_metrics = {k: gathered_metrics[k]['acc'] if 'acc' in gathered_metrics[k] else gathered_metrics[k]['mse']
                            for k in gathered_metrics}
        print(recorded_metrics)
        path = os.path.join(eval_model, 'eval.json')
        with open(path, 'w') as fw:
            json.dump(recorded_metrics, fw)
        print(f'Evaluation results are saved in {path}.')

def scoring(args, eval_model, audio_dir):
    audio_paths = glob.glob(f'{audio_dir}/*')

    sampling_rate = 16000
    device = 0

    # model
    model = prepare_model(args, reload_model=eval_model)
    model.eval()

    model.audio_tower = model.audio_tower.to(device)
    model.speech_decoder = model.speech_decoder.to(device)
    if args.model_mode == 'SpokenLLM':
        model.spoken_lm = model.spoken_lm.to(device)
    else:
        model.spoken_lm = model.spoken_lm.to('cpu')

    audio_processor = WhisperProcessor.from_pretrained(WHISPER_PATH)
    audio_tokenizer = AutoTokenizer.from_pretrained(WHISPER_PATH)
    llm_tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    processor = TasteProcessor(audio_processor, audio_tokenizer, llm_tokenizer)

    data = [
        processor(
            audio_path, sampling_rate,
            ref_audio_list=[audio_path],
        )
        for audio_path in audio_paths
    ]
    dataset = Dataset.from_list(data)

    cols = [
        'llm_token_ids', 
        'llm_token_lengths',
        'llm_word_ids',
        'audio_features',
        'audio_feature_lengths',
        'asr_token_ids',
        'asr_token_lengths',
        'asr_word_ids'
    ]

    for path, batch in zip(audio_paths, dataset):
        inputs = pad_seq_collate_fn([{k: batch[k] for k in cols}], device=device)
        loss = model.scoring(**inputs)

        json_content = {
            'path': path,
            'loss': loss.cpu().item()
        }
        json.dump(
            json_content,
            open(path + '.json', 'w'),
            ensure_ascii=False, indent=4
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--mode', default='train', type=str) 
    # options of mode: `train`, `eval`, `scoring`

    # for `eval` and `scoring`
    parser.add_argument('--eval_model', default='', type=str)

    # for `scoring`
    parser.add_argument('--audio_dir', default='', type=str)


    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'rb').read())
    for key, val in config.items():
        if not hasattr(args, key):
            setattr(args, key, val)

    if args.mode == 'train':
        train(args)
        # e.g.:
        # accelerate launch scripts/run.py --mode=train --config xxx.yaml

    elif args.mode == 'eval':
        evaluate(args, args.eval_model)
        # e.g.:
        # accelerate launch scripts/run.py --mode=eval --config xxx.yaml --eval_model xxx

    elif args.mode == 'scoring':
        scoring(args, args.eval_model, args.audio_dir)
        # e.g.:
        # python scripts/run.py --mode scoring --config xxx.yaml --eval_model xxx --audio_dir xxx
