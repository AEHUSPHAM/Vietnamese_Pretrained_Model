import datasets

raw_datasets = datasets.load_dataset('text', data_dir="/media/data/huypn10/Vietnamese_Pretrained_Model/fine-tuning/dataset/ner_preprocessed")
print(raw_datasets['train'][:10])
def split_data(examples):
        tokens = []
        labels = []
        for i in examples['text']:
            if(len(i)>0): 
                _split = i.split(" ")
                # print(_split)
                tokens.append(_split[0])
                labels.append(_split[1])
        
        examples['text'] = tokens
        examples['labels'] = labels
        return examples

raw_datasets = raw_datasets.map(split_data, batched=True)
print(raw_datasets['train'][:10])
