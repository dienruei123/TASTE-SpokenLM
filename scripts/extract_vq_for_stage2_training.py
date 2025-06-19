
import os

from datasets import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import Trainer, TrainingArguments

from taste_speech import TasteForCausalLM
from taste_speech.data.dataset import TasteDataset


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


class ExtractVQTrainer(Trainer):
    def __init__(self, add_speech_elements, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = []
        self.add_speech_elements = add_speech_elements

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        asr_indices, llm_indices = model.extract_vq(
            **{
                k: inputs[k] 
                for k in [
                    'asr_token_ids',
                    'asr_token_lengths',
                    'asr_word_ids',
                    'llm_token_ids',
                    'llm_token_lengths',
                    'llm_word_ids',
                    'audio_features',
                    'audio_feature_lengths',
                ]
            }
        )
        asr_indices = asr_indices.cpu()
        llm_indices = llm_indices.cpu()
        llm_token_ids = inputs['llm_token_ids'].cpu()
        llm_token_lengths = inputs['llm_token_lengths'].cpu()
        llm_word_ids = inputs['llm_word_ids'].cpu()
        if self.add_speech_elements:
            asr_token_ids = inputs['asr_token_ids'].cpu()
            asr_token_lengths = inputs['asr_token_lengths'].cpu()
            asr_word_ids = inputs['asr_word_ids'].cpu()
            speech_token_ids = inputs['speech_token_ids'].cpu()
            speech_token_lengths = inputs['speech_token_lengths'].cpu()
    
        for i in range(asr_indices.size(0)):
            llm_length = int(llm_token_lengths[i].item())
            result = {
                'llm_indices': llm_indices[i,:llm_length,:].unsqueeze(0).to(torch.int64),
                'llm_token_ids': llm_token_ids[i,:llm_length].unsqueeze(0).to(torch.int64),
                'llm_token_lengths': torch.tensor([llm_length], dtype=torch.int32),
                'llm_word_ids': llm_word_ids[i,:llm_length].unsqueeze(0).to(torch.int32),
            }
            if self.add_speech_elements:
                asr_length = int(asr_token_lengths[i].item())
                speech_length = int(speech_token_lengths[i].item())
                result.update({
                    'speaker_embeds': inputs['speaker_embeds'][i].unsqueeze(0).cpu(),
                    'asr_token_ids': asr_token_ids[i,:asr_length].unsqueeze(0).to(torch.int64),
                    'asr_token_lengths': torch.tensor([asr_length], dtype=torch.int32),
                    'asr_word_ids': asr_word_ids[i,:asr_length].unsqueeze(0).to(torch.int32),
                    'speech_token_ids': speech_token_ids[i,:speech_length].unsqueeze(0).to(torch.int64),
                    'speech_token_lengths': torch.tensor([speech_length], dtype=torch.int32),
                })
            self.results.append(result)
        losses = None
        logits = None
        labels = None
        return losses, logits, labels


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


def prepare_model(model_dir, attn_implementation='flash_attention_2'):
    model = TasteForCausalLM.from_pretrained_stage1(
        model_dir, 
        attn_implementation=attn_implementation,
        skip_audio_in_audio_decoder=False,
        skip_vq_in_audio_encoder=False
    )
    model.eval()
    return model


def prepare_dataset(stage1_data_folder, model_config, selected_cols):
    ds = TasteDataset(
        stage1_data_folder,
        model_config.asr_config._name_or_path,
        model_config.text_config._name_or_path,
        selected_cols=selected_cols,
    )
    return ds


def main(model_dir, output_dir, stage1_data_folder, add_speech_elements=False):
    model = prepare_model(model_dir)

    selected_cols = [
        'asr_token_ids',
        'asr_token_lengths',
        'asr_word_ids',
        'llm_token_ids',
        'llm_token_lengths',
        'llm_word_ids',
        'audio_features',
        'audio_feature_lengths',
    ]
    if add_speech_elements:
        selected_cols += [
            'speaker_embeds',
            'speech_token_ids',
            'speech_token_lengths',
        ]

    eval_dataset = prepare_dataset(stage1_data_folder, model.config, selected_cols)

    # Define training arguments
    training_args = TrainingArguments(
        do_train=True,
        do_eval=False,
        output_dir=output_dir,
        per_device_eval_batch_size=8,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_prefetch_factor=2,
        remove_unused_columns=False,
        deepspeed=None,
        no_cuda=False,
    )

    # Initialize the Trainer without the train_dataset
    trainer = ExtractVQTrainer(
        add_speech_elements=add_speech_elements,
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=pad_seq_collate_fn,
    )

    # Evaluate the model
    eval_results = trainer.evaluate()
    ds = Dataset.from_list(trainer.results)
    ds.save_to_disk(output_dir + f'/part-{LOCAL_RANK}')
    del ds

    # gathered_results = trainer.accelerator.gather(stacked_results)
    # if trainer.accelerator.is_main_process:
    #     torch.save(gathered_results, output_dir + pt_file)
    #     print('done: main')
    # else:
    #     print('done: other')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--stage1_data_folder', type=str)
    parser.add_argument('--add_speech_elements', action='store_true')
    args = parser.parse_args()

    main(args.model_dir, args.output_dir, args.stage1_data_folder, add_speech_elements=args.add_speech_elements)

    # accelerate launch scripts/extract_vq_for_stage2_training.py
