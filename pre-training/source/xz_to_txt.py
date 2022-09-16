import lzma
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str, default=None, help='Source file')
    parser.add_argument('--destination_file', type=str, default=None, help='Destination dir')
    # parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.source_file, "wt") as fout:
        with lzma.open(args.destination_file, "rt") as fin:
            for line in fin:
                fout.write(line)
