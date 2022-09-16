python ../source/pre-train_tokenizer.py \
    --tokenizer_name microsoft/deberta-v3-xsmall \
    --source_dir ../dataset/segment_cc100_1e4\
    --destination_dir ../tokenizer/spm_1e4_5e5 \
    --cache_dir ../bpe_cc100_1e4_5e5