CUDA_VISIBLE_DEVICES=0 python ../source/pre_tokenizer.py \
    --cache_dir ../cache/dataset/segment \
    --source_file /media/data/huypn10/Vietnamese_Pretrained_Model/fine-tuning/dataset/ner \
    --destination_dir ../dataset/segment_vi_cc100
