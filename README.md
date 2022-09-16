# Pre-training
## Code architecture
1. bash: bash scripts to run the pipeline
2. config: model_config (json files)
3. dataset: datasets folder (both store original **txt** dataset and the pointer to memory of datasets.load_from_disk)
4. source: main python files to run pre-training tokenizers
5. tokenizer: folder to store tokenizers
## Pre-tokenizer
- Split the original **txt** datasets into train, validation and test sets with 90%, 5%, 5%.
- Using the PyVi library to segment the datasets
- Save datasets to disk
## Pre-train_tokenizer
- Load datasets
- Train the tokenizers with SentencePiece models
- Save tokenizers 
## Pre-train_model
- Load datasets
- Load tokenizers
- Pre-train DeBERTa-v3

# Run experiments
## Preparation
- **pip install DeBERTa/.**
- **pip install transformers/.**
## Convert xz to txt
bash ../pre-training/bash/xz_to_txt.sh
## Pre-tokenizer
bash ../pre-training/bash/pre_tokenizer.sh
## Pre-train_tokenizer
bash ../pre-training/bash/pre-train_tokenizer.sh
## Pre-train_model
bash ../pre-training/bash/pre-train_model.sh
# Change log
## DeBERTa Official Github
### 1. DeBERTa -> apps -> run.py
``` python
# Old (line 259)
tokenizer = tokenizers[vocab_type](vocab_path)
# New (line 259)
if args.tokenizer_dir: 
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
else: 
    tokenizer = tokenizers[vocab_type](vocab_path)
```
### 2. DeBERTa -> apps -> tasks -> rtd_task.py
Main change direction:
- Currently use same config for generator and discriminator since the original code call config.discriminator and config.generator but i can not found the setting of those config.
- Change from read traditionally from text file to read by **datasets.load_from_disk()**
``` python
# Old 
class RTDTask()
    def __init__():
        gen_config = config.generator
        disc_config = config.discriminator
        self.config = config
        self.generator = MaskedLanguageModel(gen_config)
        self.discriminator = ReplacedTokenDetectionModel(disc_config)

        self.data_dir = data_dir

    def train_data():
        data = self.load_data(os.path.join(self.data_dir, 'train.txt'))

    def eval_data():
        ds = [self._data('dev', 'valid.txt', 'dev'),]
   
        for d in ds:
            _size = len(d.data)
            d.data = DynamicDataset(d.data, 
                feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), 
                dataset_size = _size, **kwargs)

    def _data():
        if isinstance(path, str):
            path = [path]
        data = []
        for p in path:
            input_src = os.path.join(self.data_dir, p)
            assert os.path.exists(input_src), f"{input_src} doesn't exists"
            data.extend(self.load_data(input_src))

    def load_data(self, path):
        examples = []
        with open(path, encoding='utf-8') as fs:
        for l in fs:
            if len(l) > 1:
            example = ExampleInstance(segments=[l])
            examples.append(example)
        return examples

    def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, **kwargs):
        segments = [ example.segments[0].strip().split() ]
    
# New 
class RTDTask()
    def __init__():
        # gen_config = config.generator
        # disc_config = config.discriminator
        self.config = config
        self.generator = MaskedLanguageModel(config)
        self.discriminator = ReplacedTokenDetectionModel(config)

        self.dataset = load_from_disk(args.data_dir)
        def tokenize_function(examples):
            examples["text"] = [self.tokenizer.tokenize(line.strip()) for line in examples["text"]]
            return examples

        self.dataset = self.dataset.map(tokenize_function, batched=True, num_proc=16)

    def train_data():
        data = self.load_data(self.dataset['train']['text'])

    def eval_data():
        data = self.load_data(self.dataset['validation']['text'])
        examples = ExampleSet(data)
        predict_fn = self.get_predict_fn()

        ds = [EvalData('dev', examples,
                       metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn, ignore_metric=False,
                       critial_metrics=['accuracy'])]
        for d in ds:
            _size = len(d.data)
            d.data = DynamicDataset(d.data,
                                    feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen),
                                    dataset_size=_size, **kwargs)
        return ds

    def load_data(self, path):
        return [ExampleInstance(segments=[line]) for line in dataset]

    def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, **kwargs):
        segments = [example.segments[0]]
```
## Huggingface Official Github
- modify code in transformers/src/transformers/models/deberta_v2
- Adding part_of_whole_word, _is_whitespace, _is_control, _is_punctuation
```python
def part_of_whole_word(self, token, is_bos=False):
        if is_bos:
            return True
        if (
            len(token) == 1
            and (_is_whitespace(list(token)[0]) or _is_control(list(token)[0]) or _is_punctuation(list(token)[0]))
        ) or token in ["[CLS]","[SEP]","[UNK]","[SEP]","[PAD]","[CLS]""[MASK]"]:
            return False

        word_start = b"\xe2\x96\x81".decode("utf-8")
        return not token.startswith(word_start)

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
```