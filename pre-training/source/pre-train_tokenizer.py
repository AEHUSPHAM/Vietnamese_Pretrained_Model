import datasets
import argparse
from transformers import AutoTokenizer, PreTrainedTokenizerFast,DebertaV2TokenizerFast
from tokenizers import SentencePieceBPETokenizer
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, default=None, help='Tokenizer name')
    parser.add_argument('--source_dir', type=str, default=None, help='Source file')
    parser.add_argument('--destination_dir', type=str, default=None, help='Cache dir')
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()
    #
    # # Build a tokenizer
    # bpe_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    #
    # Initialize a dataset
    raw_datasets = datasets.load_from_disk(args.source_dir)
    # raw_datasets = datasets.load_dataset('text', data_files = '/media/data/huypn10/Efficient_Vietnamese_BERT/pre-training/dataset/cc100_1e4.txt')
    # Build an iterator over this dataset
    def get_training_corpus():
        dataset = raw_datasets["train"]
        for start_idx in range(0, len(dataset), 1000):
            samples = dataset[start_idx : start_idx + 1000]
            yield samples["text"]

    training_corpus = get_training_corpus()
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
    tk_tokenizer = SentencePieceBPETokenizer()
    # tk_tokenizer=AutoTokenizer.from_pretrained('microsoft/deberta-v3-small', use_fast=False)
    tk_tokenizer.train_from_iterator(
        training_corpus,
        vocab_size=128000,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens
    )
    #
    # # convert
    tokenizer = DebertaV2TokenizerFast(tokenizer_object=tk_tokenizer, model_max_length=128, special_tokens=special_tokens)
    tokenizer.bos_token = "<s>"
    tokenizer.bos_token_id = tk_tokenizer.token_to_id("<s>")
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = tk_tokenizer.token_to_id("<pad>")
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = tk_tokenizer.token_to_id("</s>")
    tokenizer.unk_token = "<unk>"
    tokenizer.unk_token_id = tk_tokenizer.token_to_id("<unk>")
    tokenizer.cls_token = "<cls>"
    tokenizer.cls_token_id = tk_tokenizer.token_to_id("<cls>")
    tokenizer.sep_token = "<sep>"
    tokenizer.sep_token_id = tk_tokenizer.token_to_id("<sep>")
    tokenizer.mask_token = "<mask>"
    tokenizer.mask_token_id = tk_tokenizer.token_to_id("<mask>")
    # # and save for later!
    tokenizer.save_pretrained(args.destination_dir)
    tk_tokenizer.save(args.destination_dir + '/spm.model')
    