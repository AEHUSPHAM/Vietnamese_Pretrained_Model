from pyvi import ViTokenizer
import time
import datasets
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str, default=None, help='Source file')
    parser.add_argument('--destination_dir', type=str, default=None, help='Destination dir')
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_path = args.source_file
    start_time = time.perf_counter()

    dataset = datasets.load_dataset("text", data_files=data_path, cache_dir=args.cache_dir)
    dataset['validation'] = datasets.load_dataset("text",
                                                  data_files=data_path,
                                                  split=f"train[:5%]",
                                                  cache_dir=args.cache_dir)
    dataset['test'] = datasets.load_dataset("text",
                                            data_files=data_path,
                                            split=f"train[5%:10%]",
                                            cache_dir=args.cache_dir)
    dataset['train'] = datasets.load_dataset("text",
                                             data_files=data_path,
                                             split=f"train[10%:]",
                                             cache_dir=args.cache_dir)

    def tokenize_function(examples):
        examples["text"] = [ViTokenizer.tokenize(line) for line in examples["text"]]
        return examples

    dataset = dataset.map(tokenize_function, batched=True, num_proc=16)
    dataset.save_to_disk(args.destination_dir)
    print(dataset)
    finish_time = time.perf_counter()
    print(f"Finished in {finish_time - start_time} seconds")

