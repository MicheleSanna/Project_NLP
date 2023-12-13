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

if __name__=='__main__':
    main()
