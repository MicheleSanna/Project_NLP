import torch
import torch.nn as nn 
from models import * 
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from test import *
from JsonDatasets import * 
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim.lr_scheduler import StepLR

N_STEP = 25
TRAIN_DATASET = 20000
BATCH_SIZE = 16
N_FILE = 1

class LinearProbes:

    #INITIALIZING FUNCTIONS
    #These internal functions are used tu initialize the trainer object. They set
    #the environment, the model and freeze half of it

    def __init__(self, model, loss_function, epochs, dataloader, device, clip_value=2):
        self.epoch = 0
        self.writer= SummaryWriter()
        self.device = device
        #self.global_rank = int(os.environ["RANK"])
        self.loss_function = loss_function
        self.num_epochs = epochs
        self.dataloader = dataloader
        self.clip_value = clip_value
        self._init_model(model, device)
        self._create_linear_probes()

    def _create_linear_probes(self):
        self.linear_probes = [nn.Linear(768, 2) for i in range(13)]
        self.optimizers = [torch.optim.AdamW(self.linear_probes[i].parameters(),lr=5e-5,eps=1e-8) for i in range(13)]
        self.schedulers = [StepLR(self.optimizers[i], 8, gamma=0.1) for i in range(13)]
        for probe in self.linear_probes:
            probe.to(self.device)

    def _init_model(self, model, device):
        self.model = model
        self.model.to(device)
        self._freeze_model()
        #if os.path.exists("saves/checkpoint.save"):
        #    self._load()

    def _freeze_model(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    #TRAINING FUNCTIONS
    #These functions are used to train the model 
    #---------------------------------------------------

    def train(self):
        #self.evaluate(-1, self._accuracy)
        #check_parameters_grad(self.model)
        last_epoch = 0
        for epoch in range (self.epoch, self.num_epochs):
            print("EPOCH {epoch}\n--------------".format(epoch=epoch))
            self._do_epoch(epoch)
            #self.evaluate(epoch, self._accuracy)
            for i in range(13):
                self.schedulers[i].step()
            self.writer.close()
            last_epoch = epoch
        self.evaluate(last_epoch, self._accuracy)
        
        return self.model
    
    def _do_epoch(self, epoch):
        self.model.eval()
        #self.dataloader.sampler.set_epoch(epoch)
        running_losses = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i in range(13):
            self.linear_probes[i].train()
        for step, batch in enumerate(self.dataloader): 
            batch_labels, batch_inputs = self._decompose_batch(batch)
            batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            self.model.zero_grad()
            with torch.no_grad():
                outputs = self.model(**batch_inputs)['hidden_states']
            for i in range(13):
                running_losses[i] += self._train_probe(outputs[i], batch_labels, i)

            if (step+1)%N_STEP == 0:
                for i in range(13):
                    print("Step[{step}] | Loss at layer {i}[{loss}] | Lr{lr}".format(step=step + 1, i=i, loss=running_losses[i]/N_STEP, lr=self.schedulers[i].get_last_lr()))
                    self.writer.add_scalar("Loss epoch {epoch} at layer {i}".format(epoch=epoch, i=i), running_losses[i]/N_STEP, epoch*TRAIN_DATASET+step*BATCH_SIZE)
                    running_losses[i] = 0
                print("-------------------------------------------------------------------------------------------------")
                    
                #print("Labels: ")
                #print(batch_labels)
                #print("Output: ")
                #print(outputs.squeeze())

    def _decompose_batch(self, batch):
        batch_labels = batch[0]
        batch_inputs = batch[1]
        batch_inputs['input_ids'] = torch.squeeze(batch_inputs['input_ids'], dim=1)
        batch_inputs['token_type_ids'] = torch.squeeze(batch_inputs['token_type_ids'], dim=1)
        batch_inputs['attention_mask'] = torch.squeeze(batch_inputs['attention_mask'], dim=1)
        return batch_labels, batch_inputs   

    def _train_probe(self, outputs, labels, i):
        self.linear_probes[i].zero_grad()
        preds = self.linear_probes[i](outputs[:,0,:])
        loss = self.loss_function(preds, labels)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.clip_value)
        self.schedulers[i].optimizer.step()
        return loss
    
    #EVALUATE FUNCTIONS 
    #These functions are used to evaluate the model
    #------------------------------------------------------------
    
    def evaluate(self, epoch, metric): 
        test_dataset = JsonFastClassificationDataset(4000, "HighPerplexityDataset_gpt2_test.", N_FILE, self.dataloader.dataset.tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)
        self.model.eval()
        
        for i in range(13):
            test_loss, test_metric = [], []
            self.linear_probes[i].eval()
            for step, batch in enumerate(test_dataloader):
                batch_labels, batch_inputs = self._decompose_batch(batch)
                batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                with torch.no_grad(): 
                    outputs = self.model(**batch_inputs)['hidden_states']
                    preds = self.linear_probes[i](outputs[i][:,0,:])
                    loss = self.loss_function(preds, batch_labels)
                    test_loss.append(loss.item())
                    metric_value = metric(preds, batch_labels)
                    test_metric.append(metric_value.item())
            print("Layer {i}: Loss[{loss}] | Accuracy[{metric}]".format(i=i, loss=sum(test_loss)/len(test_loss), metric=sum(test_metric)/len(test_metric)))
                    
    def _accuracy(self, outputs, labels):
        preds = torch.argmax(outputs, dim=1)
        corrects = torch.sum(preds == labels)
        return torch.sum(corrects)/preds.size(dim=0)
