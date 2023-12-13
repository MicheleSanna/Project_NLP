import torch
import json 
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer
from transformers import BertForMaskedLM, BertTokenizer

model = BertForMaskedLM.from_pretrained("bert-base-cased")
model.to(0)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

def ppl_gpt2(model, inputs, tokenizer):
    inputs = tokenizer(inputs, return_tensors="pt")
    inputs.to(0)
    with torch.no_grad():
        output = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"])
        ppl = torch.exp(output.loss)
    return ppl.item()

def ppl_bert(model, inputs, tokenizer):
    tensor_input = tokenizer.encode(inputs, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    len = masked_input.size(dim=0)
    masked_input = masked_input.to(0)
    labels = labels.to(0)
    loss = 0
    with torch.inference_mode():
        for i in range(len):
            output = model(torch.unsqueeze(masked_input[i, :], 0), labels=torch.unsqueeze(labels[i, :], 0))
            loss += output.loss
    ppl = torch.exp(loss/len)
    return ppl.item()

#IMPORTANT: ppl_bert and ppl_bert_2 are equivalent. They do the same calculation but 
#ppl_bert is less VRAM intensive (in order to run on my beautiful laptop) and ppl_bert_2
#is much faster but at the cost of more VRAM

def ppl_bert_2(model, inputs, tokenizer):
    with torch.no_grad():
        tensor_input = tokenizer.encode(inputs, return_tensors='pt')
        repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
        masked_input = masked_input.to(0)
        labels = labels.to(0)
        loss = model(masked_input, labels=labels).loss
        ppl = torch.exp(loss)
    return ppl.item()

def conditions(sample):
    if len(sample['reviewText']) < 510 and len(sample['reviewText']) > 200:
        ppl = ppl_bert_2(model, sample['reviewText'], tokenizer)
        if ppl >= 14.28:
            return True
    return False

def makePerplexityDataset (model, tokenizer, device, ppl_fun, file, name):
    model.to(device)
    model.eval()
    i = 0
    with open(file) as f:
        for line in f:
            sample = json.loads(line)
            newline={'overall': ppl_fun(model, sample['reviewText'], tokenizer), 'reviewText': sample['reviewText']}
            with open(name, 'a') as f:
                print(json.dumps(newline), file=f)
            i+=1
            if i%100 == 0:
                print(i)
            if i >= 10000:
                break

def UniqueFile(dim, file, name):
    n1star = 0
    n5star = 0
    n = 0
    printFlag = False
    eccezioni = 0
    testdim = dim/5
    n1star_test = 0
    n5star_test = 0
    with open(file) as f: 
        for line in f:
            if n1star + n5star <= dim:
                sample = json.loads(line)
                printFlag = False
                if sample['overall'] == 1.0 and n1star < dim/2:
                    if conditions(sample):
                        n1star = n1star+1
                        printFlag = True
                if sample['overall'] == 5.0 and n5star < dim/2:
                    if conditions(sample):
                        n5star = n5star+1   
                        printFlag = True         

                if printFlag == True:
                    sample_str = json.dumps(sample)
                    with open(name, 'a') as f:
                        print(sample_str, file=f)
                    if n%100 == 0:
                        print("Step: ", n)
                        print("1.0: " + str(n1star) +  " | 5.0: " + str(n5star))
                    n = n+1

            if n1star + n5star >= dim:
                sample = json.loads(line)
                printFlag = False
                if sample['overall'] == 1.0 and n1star_test < testdim/2:
                    if conditions(sample):
                        n1star_test = n1star_test+1
                        printFlag = True
                if sample['overall'] == 5.0 and n5star_test < testdim/2:
                    if conditions(sample):
                        n5star_test = n5star_test+1   
                        printFlag = True         

                if printFlag == True:
                    sample_str = json.dumps(sample)
                    with open(name + "_test", 'a') as f:
                        print(sample_str, file=f)
                    if n%100 == 0:
                        print("Step: ", n)
                        print("TESTFILE: " + "1.0: " + str(n1star_test) +  " | 5.0: " + str(n5star_test))
                    n = n+1
            
            if (n1star + n5star) >= dim and (n1star_test + n5star_test) >=testdim:
                print("UNA STELLA: ", n1star_test)
                print("CINQUE STELLE: ", n5star_test)
                break
    print("UNA STELLA: ", n1star)
    print("CINQUE STELLE: ", n5star)

def isMedian(dataset, median):
    dataloader = DataLoader(dataset, batch_size=1)
    count = 0
    for step, batch in enumerate(dataloader): 
        batch_ppl = batch[0].item()
        if median < batch_ppl:
            count +=1
    print(count)

print("START")
UniqueFile(20000, "TenMillions.json", "HighPerplexityDataset_bert.json")
