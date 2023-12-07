import json
from torch.utils.data import IterableDataset, Dataset

MAXDIM = 5000000
class JsonDataset(IterableDataset):
    def __init__(self, file, tokenizer):
        self.file = file
        self.a = 0
        self.tokenizer = tokenizer

    def __iter__(self):
            self.a = self.a+1
            with open(self.file) as f:
                for sample_line in f:
                    sample = json.loads(sample_line)
                    yield (sample['overall']-1)*0.25, self.tokenizer(sample['reviewText'], add_special_tokens=True, padding='max_length', max_length=500, truncation='longest_first',return_tensors='pt')

class JsonFastDataset(Dataset):
    def __init__(self,file, tokenizer):
        self.data = []
        self.i = 0
        self.tokenizer = tokenizer
        
        with open(file) as f:
            for line in f:
                if(self.i%10000 == 0):
                    print("Riga:", self.i)
                self.i = self.i+1
                sample = json.loads(line)
                self.data.append(((sample['overall']-1)*0.25, self.tokenizer(sample['reviewText'], add_special_tokens=True, padding='max_length', max_length=500, truncation='longest_first',return_tensors='pt')))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]    

class JsonFastRandomDataset(Dataset):
    def __init__(self, dim : int, filename, nFile: int, tokenizer):
        self.filename = filename
        self.dim = dim
        self.nFile = nFile
        self.tokenizer = tokenizer

    def __len__(self):
        return self.dim
    
    def __getitem__(self, idx):
        place : int = 0
        with open(self.filename + str(int(idx//(MAXDIM/self.nFile))) + ".json", 'r') as f:
            for line in f:
                if place != idx%(MAXDIM/self.nFile):
                    place = place +1
                else:
                    sample = json.loads(line)
                    return (sample['overall']-1)*0.25, self.tokenizer(sample['reviewText'], add_special_tokens=True, padding='max_length', max_length=500, truncation='longest_first',return_tensors='pt')

class JsonFastClassificationDataset(Dataset):
    def __init__(self, dim : int, filename, nFile: int, tokenizer):
        self.filename = filename
        self.dim = dim
        self.nFile = nFile
        self.tokenizer = tokenizer

    def __len__(self):
        return self.dim
    
    def __getitem__(self, idx):
        place : int = 0
        with open(self.filename + str(int(idx//(MAXDIM/self.nFile))) + ".json", 'r') as f:
            for line in f:
                if place != idx%(MAXDIM/self.nFile):
                    place = place +1
                else:
                    sample = json.loads(line)
                    return int(sample['overall']) - 1, self.tokenizer(sample['reviewText'], add_special_tokens=True, padding='max_length', max_length=500, truncation='longest_first',return_tensors='pt')
