import os

data_root = "/proj/mtklmadm/data/speech/EXPRESSO/expresso"
transcripts_fpath = "/proj/mtklmadm/data/speech/EXPRESSO/expresso/read_transcriptions.txt"
# split_fid_fpath = "/proj/mtklmadm/data/speech/EXPRESSO/expresso/splits/test.txt"
split_fid_fpath = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/eval/expresso-singing.txt"

_normal_classes = ['happy', 'default', 'sad', 'enunciated', 'confused']
_special_classes = ['laughing', 'whisper']
_singing_classes = ['singing']
target_classes = _singing_classes
target_split_name = "test-singing"

fid_to_trans = {}
with open(transcripts_fpath, 'r') as fr:
    for l in fr:
        fid, trans = l.strip().split('\t')
        fid_to_trans[fid] = trans

fids_with_trans = list(fid_to_trans.keys())
valid_fid_and_fpaths = []
with open(split_fid_fpath, 'r') as fr:
    for l in fr:
        fid = l.strip().split('\t')[0]
        items = fid.split('_')
        if len(items) != 3 or '-' in fid: continue
        _shard, _cls, _id = items
        assert fid in fids_with_trans, f"fid={fid} not found in transcripts!"
        if _cls not in target_classes: continue
        fpath = os.path.join(data_root, "audio_48khz/read", _shard, _cls, "base", f"{fid}.wav")
        assert os.path.exists(fpath), f"{fpath} not found."
        valid_fid_and_fpaths.append((fid, fpath))

output_fpath = f"/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/eval/test-audio_expresso-{target_split_name}.tsv"
with open(output_fpath, 'w') as fw:
    for fid, fpath in valid_fid_and_fpaths:
        trans = fid_to_trans[fid]
        fw.write(f"{fpath}\t{trans}\n")

