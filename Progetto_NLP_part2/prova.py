import torch
import torch.nn as nn 
#from perplexity_dataset import *
from models import *
from test import *
from JsonDatasets import *
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from transformers import GPT2ForSequenceClassification, BertForSequenceClassification
from linear_probes import LinearProbes
from transformers import GPT2Tokenizer, BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#Percentili:
#12.5° : 27.03
#25°   : 33.02
#37.5° : 36.525
#50°   : 45.805
#62.5° : 53.84
#75°   : 65.655
#87.5° : 87.45

activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def set_up_arguments(epochs, model, loss, device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    #tokenizer.pad_token = tokenizer.eos_token
    dataset = JsonFastClassificationDataset(20000, "LowPerplexityDataset_bert.", 1, tokenizer)#9999018
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    args = {'model': model,
            'epochs': epochs,
            'loss_function': loss,
            'dataloader': dataloader,
            'clip_value': 2,
            'device': device}
    
    return args, tokenizer

#------------------------------------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################
#------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    model = BertForSequenceClassification.from_pretrained("bert-base-cased", output_hidden_states=True, num_labels=2)
    model.eval()

    args, tokenizer = set_up_arguments(10,model, nn.CrossEntropyLoss(), 0)

    linear_probes=LinearProbes(**args)
    linear_probes.train()



def other():
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", output_hidden_states=True, num_labels=2)
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
    print(model)

    inputs = tokenizer("The pen is on the", return_tensors = "pt")

    #output_info(outputs)

    #makePerplexityDataset(model, tokenizer, torch.device("cuda"), ppl_gpt2, "FiveMillions.json", "PerplexityDataset.json")
    #print(findMedian(42, -1, 100000, JsonDataset("PerplexityDataset.json", tokenizer)))
    #isMedian(JsonDataset("PerplexityDataset.json", tokenizer), 27.03)


    print("--------------------------------------------------------")
    for name, param in model.named_parameters():
        print("{name} | {size}".format(name = name, size = param.size()))


    fun = torch.nn.CosineSimilarity(dim = 2)
    softmax = torch.nn.Softmax(dim=2)
    writer = SummaryWriter()
    #model2 = Prova(model)
    #writer.add_graph(model2, inputs["input_ids"])
    #writer.close()
    layer = 11
    lm_head = model.lm_head
    l1 = model.transformer.h[layer].ln_1.register_forward_hook(getActivation('ln_1'))
    l2 = model.transformer.h[layer].attn.c_proj.register_forward_hook(getActivation('attn.c_proj'))
    l3 = model.transformer.h[layer].attn.resid_dropout.register_forward_hook(getActivation('attn.resid_dropout'))
    l4 = model.transformer.h[layer].ln_2.register_forward_hook(getActivation('ln_2'))
    l5 = model.transformer.h[layer].mlp.c_proj.register_forward_hook(getActivation('mlp.c_proj'))
    l6 = model.transformer.h[layer].mlp.dropout.register_forward_hook(getActivation('mlp.dropout'))
    l7 = model.transformer.ln_f.register_forward_hook(getActivation('ln_f'))
    layernorm = model.transformer.ln_f


    outputs = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"])
    print("--------------------------")
    for key in activation.keys():
        print("{key}: {size}".format(key=key, size=activation[key].size()))


    for key in activation.keys():
        print("--------------------------")
        print(key)
        for i in range(13):
            print("Difference at layer {i}, key {key}: {x}".format(i = i, key=key, x = fun(outputs.hidden_states[i], activation[key])))

    #for name, param in lm_head.named_parameters():
    #    print(name)
    #    print(param.size())

    print(fun(outputs.logits, lm_head(outputs.hidden_states[1])).size())
    for i in range(13):
        print("Difference at layer {i}: {x}".format(i = i, x = fun(layernorm(outputs.hidden_states[i]), outputs.hidden_states[12]), dim=2))

    model2 = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=5, output_hidden_states=True)
    model2.eval()
    print(model2)    
    outputs = model2(inputs["input_ids"])
    score = model2.score
    print(outputs.hidden_states[12][:,-1,:])
    print(score(outputs.hidden_states[12])[:,-1,:])
    print(outputs.logits)

if __name__=='__main__':
    main()
