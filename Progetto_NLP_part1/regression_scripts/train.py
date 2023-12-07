from conf import * 
from models import *
from test import *
from trainer import *
from JsonDatasets import * 
from contextlib import closing
import socket
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import BertTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Config
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os
import sys


def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def device_chooser():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    return device

def init_distributed(rank, dist_url):
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
     # default
    
    # only works with torch.distributed.launch // torch.run
    #os.environ["MASTER_ADDR"] = MASTER_ADDR
    #os.environ["MASTER_PORT"] = MASTER_PORT
    dist.init_process_group(backend=BACKEND, init_method=dist_url, world_size=int(os.environ["WORLD_SIZE"]), rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"])) 

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

def set_up_arguments(epochs, model, loss, device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    dataset = JsonFastRandomDataset(TRAIN_DATASET, "scattered_dataset/ScatteredTenMillions", N_FILE, tokenizer)#9999018
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=DistributedSampler(dataset=dataset, shuffle=True), num_workers=NUM_WORKERS)
    scheduler = StepLR(torch.optim.AdamW(model.parameters(),lr=5e-4,eps=1e-8), 1, gamma=0.2) 

    args = {'model': model,
            'epochs': epochs,
            'scheduler': scheduler,
            'loss_function': loss,
            'dataloader': dataloader,
            'clip_value': 2,
            'device': device}
    
    return args, tokenizer






#print(GPT2LMHeadModel)
#print(GPT2Config())


def main(rank: int):
    print("My rank is: ", rank)
    init_distributed(rank, "env://")
    args, tokenizer = set_up_arguments(NUM_EPOCHS, BertForRegressionSigmoid(), nn.MSELoss(), int(os.environ["LOCAL_RANK"]))
    if TESTING:
        bert_test(args['model'], tokenizer)
    
    print("------------------------\n")

    if TESTING:
        test_dataloader(args['dataloader'])

    trainer = CustomTrainer(**args)
    model = trainer.train()
    torch.save(model, "Final_model.pth")
    destroy_process_group()


if __name__ == '__main__':
    world_size=torch.cuda.device_count()
    #port = get_open_port()
    #os.environ["MASTER_PORT"] = str(53191)
    print("PORT: ", os.environ["MASTER_PORT"])
    print("WORLD SIZE: ", os.environ["WORLD_SIZE"])
    print("MASTER NODE: ", os.environ["MASTER_ADDR"])
    #print("Visible devices: ", os.environ["CUDA_VISIBLE_DEVICES"])
    
    print("My slurm id is: ", int(os.environ['SLURM_PROCID']))
    main(int(os.environ["RANK"]))