import torch
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from conf import * 
from test import *
from JsonDatasets import * 
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.utils.data.distributed import DistributedSampler
import os

class CustomTrainer:
    def __init__(self, model, scheduler, loss_function, epochs, dataloader, device, clip_value=2):
        self.epoch = 0
        self.writer= SummaryWriter()
        self.device = device
        self.global_rank = int(os.environ["RANK"])
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.num_epochs = epochs
        self.dataloader = dataloader
        self.clip_value = clip_value
        self._init_model(model, device)
        print("I'm process {rank} using GPU {local_rank}".format(rank=self.global_rank, local_rank=self.device))
        
    def _init_model(self, model, device):
        self.model = model
        self.model.to(device)
        self._freeze_half()
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[device])#device_ids=int(os.environ['LOCAL_RANK']))
        if os.path.exists("saves/checkpoint.save"):
            self._load()
    
    def _freeze_half(self):
        for name, param in self.model.bert.embeddings.named_parameters():
            param.requires_grad = False
        for name, param in self.model.bert.encoder.named_parameters():
            if 'layer.0.' in name:
                param.requires_grad = False
            if 'layer.1.' in name:
                param.requires_grad = False
            if 'layer.2.' in name:
                param.requires_grad = False
            if 'layer.3.' in name:
                param.requires_grad = False
            if 'layer.4.' in name:
                param.requires_grad = False
            if 'layer.5.' in name:
                param.requires_grad = False
        
    def _decompose_batch(self, batch):
        batch_labels = batch[0]
        batch_inputs = batch[1]
        batch_inputs['input_ids'] = torch.squeeze(batch_inputs['input_ids'], dim=1)
        batch_inputs['token_type_ids'] = torch.squeeze(batch_inputs['token_type_ids'], dim=1)
        batch_inputs['attention_mask'] = torch.squeeze(batch_inputs['attention_mask'], dim=1)
        batch_labels = batch_labels.to(torch.float32)
        return batch_labels, batch_inputs

    def _do_epoch(self, epoch):
        self.model.train()
        running_loss = 0
        self.dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(self.dataloader): 
            batch_labels, batch_inputs = self._decompose_batch(batch)
            batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.model.zero_grad()
            outputs = self.model(**batch_inputs) 
            loss = self.loss_function(outputs.squeeze(), batch_labels)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.scheduler.optimizer.step()
            running_loss += loss

            if (step+1)%N_STEP == 0:
                print("Step[{step}] | Loss[{loss}] | Lr{lr}".format(step=step + 1, loss=loss, lr=self.scheduler.get_last_lr()))
                #print("Labels: ")
                #print(batch_labels)
                #print("Output: ")
                #print(outputs.squeeze())
                self.writer.add_scalar("Loss epoch " + str(epoch), running_loss/N_STEP, epoch*TRAIN_DATASET+step*BATCH_SIZE)
                running_loss=0

    def train(self):
        self.evaluate(self.epoch-1)
        check_parameters_grad(self.model)
        for epoch in range (self.epoch, self.num_epochs):
            print("EPOCH {epoch}\n--------------".format(epoch=epoch))
            self._do_epoch(epoch)
            self.evaluate(epoch)
            self.scheduler.step()
            self.writer.close()
            if self.global_rank == 0:
                self._save(epoch+1)     
        return self.model
                    
    def _r2_score(self, outputs, labels):
        labels_mean = torch.mean(labels)
        ss_tot = torch.sum((labels - labels_mean) ** 2)
        ss_res = torch.sum((labels - outputs) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2

    def evaluate(self, epoch): 
        test_dataset = JsonFastRandomDataset(EVALUATE_DATASET, "scattered_test/ScatteredTenMillions_test", N_FILE, BertTokenizer.from_pretrained('bert-base-cased'))
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=DistributedSampler(dataset=test_dataset, shuffle=True), num_workers=NUM_WORKERS)
        self.model.eval()
        test_loss, test_r2, stupid_loss, stupid_r2 = [], [], [], []
        stupid = torch.full((torch.Size([BATCH_SIZE])), 0.5).to(self.device)
        for step, batch in enumerate(test_dataloader):
            batch_labels, batch_inputs = self._decompose_batch(batch)
            batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            with torch.no_grad(): 
                outputs = self.model(**batch_inputs) 
                #print("Inputs: ", batch_inputs)
                outputs = outputs.squeeze()
                loss = self.loss_function(outputs, batch_labels)
                st_loss = self.loss_function(stupid, batch_labels)
                test_loss.append(loss.item())
                stupid_loss.append(st_loss.item())
                if step%512 == 0:
                    print("Labels: ", batch_labels)
                    print("Stupid: ", stupid)
                    print("Outputs: ", outputs)
                    print("------------------------")

                r2 = self._r2_score(outputs, batch_labels)
                st_r2 = self._r2_score(stupid, batch_labels)
                test_r2.append(r2.item())
                stupid_r2.append(st_r2.item())
            if step >= EVALUATE_STEPS: 
                break
        
        self.writer.add_scalar("Test loss", sum(test_loss)/len(test_loss), epoch)
        self.writer.add_scalar("Test r^2", sum(test_r2)/len(test_r2), epoch)
        
        print("Mean loss[{loss}] | Mean r^2[{r2}]".format(loss= sum(test_loss)/len(test_loss), r2=sum(test_r2)/len(test_r2)))
        print("Stupid loss[{loss}] | Stupid r^2[{r2}]".format(loss= sum(stupid_loss)/len(stupid_loss), r2=sum(stupid_r2)/len(stupid_r2)))
    
    def _save(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.scheduler.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()}, "saves/checkpoint.save")
        torch.save(self.model, "saves/sentiment_model_" + str(epoch) + ".pth")

    def _load(self):
        print("Loading checkpoint...")
        checkpoint = torch.load("saves/checkpoint.save")
        print("Retrieving epoch...")
        self.epoch = checkpoint['epoch']
        print("Loading model state...")
        self.model.module.load_state_dict(checkpoint['model_state_dict'])#torch.load("saves/sentiment_model_" + str(self.epoch) + ".mod")
        print("Loading scheduler state...")
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Loading optmizer state...")
        self.scheduler.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("LOADED!")
