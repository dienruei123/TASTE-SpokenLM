import argparse

from taste_speech import TasteConfig, TasteForCausalLM


def main(args):
    config = TasteConfig.from_pretrained(args.model_config)
    model = TasteForCausalLM(config)
    model.save_pretrained(args.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='configs/model/taslm.json', type=str)
    parser.add_argument('--model_dir', default='storage/exp/TASLM-SEED/', type=str) 

    args = parser.parse_args()
    main(args)
