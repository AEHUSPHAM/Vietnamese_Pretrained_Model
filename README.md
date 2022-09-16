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
- [Clone DeBERTa github repository](https://github.com/microsoft/DeBERTa.git)
- [Download modified folder](https://drive.google.com/drive/folders/10Cr5cp2b0dU0ufgpj165Tql2lf9Vm9T7?usp=sharing) 
- Replace the folder DeBERTa/app with the download folder
- **pip install DeBERTa/.**
- **git clone https://github.com/huggingface/transformers**
- go to modify code in transformers/src/transformers/models/deberta_v2 and modify the content like the below
- Then **pip install transformers/.**
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
- Whole file:
```python
# coding=utf-8
# Copyright 2020 Microsoft and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast Tokenization class for model DeBERTa."""

import os
from shutil import copyfile
from typing import Optional, Tuple

from ...file_utils import is_sentencepiece_available
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


if is_sentencepiece_available():
    from .tokenization_deberta_v2 import DebertaV2Tokenizer
else:
    DebertaV2Tokenizer = None
import unicodedata
logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spm.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/spm.model"
        ),
        "microsoft/deberta-v2-xxlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/spm.model"
        ),
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-v2-xlarge": 512,
    "microsoft/deberta-v2-xxlarge": 512,
    "microsoft/deberta-v2-xlarge-mnli": 512,
    "microsoft/deberta-v2-xxlarge-mnli": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-v2-xlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xlarge-mnli": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge-mnli": {"do_lower_case": False},
}


class DebertaV2TokenizerFast(PreTrainedTokenizerFast):
    r"""
    Constructs a DeBERTa-v2 fast tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        bos_token (`string`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.
            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.
        eos_token (`string`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token. When building a sequence using special tokens, this is not the token that is
            used for the end of sequence. The token used is the `sep_token`.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = DebertaV2Tokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=False,
        split_by_punct=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ) -> None:
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            split_by_punct=split_by_punct,
            **kwargs,
        )

        self.do_lower_case = do_lower_case
        self.split_by_punct = split_by_punct
        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

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

    def set_save_slow(self):
        self.can_save_slow_tokenizer = True

    def printf(self):
        print(self.vocab_file)


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

```# Vietnamese_Pretrained_Model# Pre-training
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
- [Clone DeBERTa github repository](https://github.com/microsoft/DeBERTa.git)
- [Download modified folder](https://drive.google.com/drive/folders/10Cr5cp2b0dU0ufgpj165Tql2lf9Vm9T7?usp=sharing) 
- Replace the folder DeBERTa/app with the download folder
- **pip install DeBERTa/.**
- **git clone https://github.com/huggingface/transformers**
- go to modify code in transformers/src/transformers/models/deberta_v2 and modify the content like the below
- Then **pip install transformers/.**
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
- Whole file:
```python
# coding=utf-8
# Copyright 2020 Microsoft and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast Tokenization class for model DeBERTa."""

import os
from shutil import copyfile
from typing import Optional, Tuple

from ...file_utils import is_sentencepiece_available
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


if is_sentencepiece_available():
    from .tokenization_deberta_v2 import DebertaV2Tokenizer
else:
    DebertaV2Tokenizer = None
import unicodedata
logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spm.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/spm.model"
        ),
        "microsoft/deberta-v2-xxlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/spm.model"
        ),
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-v2-xlarge": 512,
    "microsoft/deberta-v2-xxlarge": 512,
    "microsoft/deberta-v2-xlarge-mnli": 512,
    "microsoft/deberta-v2-xxlarge-mnli": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-v2-xlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xlarge-mnli": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge-mnli": {"do_lower_case": False},
}


class DebertaV2TokenizerFast(PreTrainedTokenizerFast):
    r"""
    Constructs a DeBERTa-v2 fast tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        bos_token (`string`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.
            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.
        eos_token (`string`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token. When building a sequence using special tokens, this is not the token that is
            used for the end of sequence. The token used is the `sep_token`.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = DebertaV2Tokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=False,
        split_by_punct=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ) -> None:
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            split_by_punct=split_by_punct,
            **kwargs,
        )

        self.do_lower_case = do_lower_case
        self.split_by_punct = split_by_punct
        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

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

    def set_save_slow(self):
        self.can_save_slow_tokenizer = True

    def printf(self):
        print(self.vocab_file)


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