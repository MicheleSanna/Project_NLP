import torch
import torch.nn as nn 
from models import * 
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

    #INITIALIZING FUNCTIONS
    #These internal functions are used tu initialize the trainer object. They set
    #the environment, the model and freeze half of it

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
        self.model.config.pad_token_id = model.config.eos_token_id
        self.model.to(device)
        self._freeze_half_gpt()
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[device])#device_ids=int(os.environ['LOCAL_RANK']))
        if os.path.exists("saves/checkpoint.save"):
            self._load()
    
    def _freeze_half_bert(self):
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
    
    def _freeze_half_gpt(self):
        for name, param in self.model.transformer.named_parameters():
            if 'wte.weight' in name:
                param.requires_grad = False
            if 'wpe.weight' in name:
                param.requires_grad = False
            if 'h.0.' in name:
                param.requires_grad = False
            if 'h.1.' in name:
                param.requires_grad = False
            if 'h.2.' in name:
                param.requires_grad = False
            if 'h.3.' in name:
                param.requires_grad = False
            if 'h.4.' in name:
                param.requires_grad = False
            if 'h.5.' in name:
                param.requires_grad = False

    #TRAINING FUNCTIONS
    #These functions are used to train the model 
    #---------------------------------------------------
     
    def train(self):
        self.evaluate(self.epoch-1, self._accuracy)
        check_parameters_grad(self.model)
        for epoch in range (self.epoch, self.num_epochs):
            print("EPOCH {epoch}\n--------------".format(epoch=epoch))
            self._do_epoch(epoch)
            self.evaluate(epoch, self._accuracy)
            #self.scheduler.step()
            self.writer.close()
            #if self.global_rank == 0:
                #compare_model_param(self.model.module, BertForClassification(5), self.device)
            if self.global_rank == 0:
                self._save(epoch+1)     
        return self.model
    
    def _decompose_batch(self, batch):
        batch_labels = batch[0]
        batch_inputs = batch[1]
        batch_inputs['input_ids'] = torch.squeeze(batch_inputs['input_ids'], dim=1)
        #batch_inputs['token_type_ids'] = torch.squeeze(batch_inputs['token_type_ids'], dim=1)
        batch_inputs['attention_mask'] = torch.squeeze(batch_inputs['attention_mask'], dim=1)
        #batch_labels = batch_labels.to(torch.float32)
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
            outputs = self.model(**batch_inputs)['logits']
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
                self.scheduler.step()

    #EVALUATE FUNCTIONS
    #These functions are used to evaluate the model
    #------------------------------------------------------------

    def evaluate(self, epoch, metric): 
        test_dataset = JsonFastClassificationDataset(EVALUATE_DATASET, "scattered_test/ScatteredTenMillions_test", N_FILE, self.dataloader.dataset.tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=DistributedSampler(dataset=test_dataset, shuffle=True), num_workers=NUM_WORKERS)
        self.model.eval()
        test_loss, test_metric, naive_loss, naive_metric = [], [], [], []
        stupid = torch.full((torch.Size([BATCH_SIZE])), 0.5).to(self.device)
        for step, batch in enumerate(test_dataloader):
            batch_labels, batch_inputs = self._decompose_batch(batch)
            batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            with torch.no_grad(): 
                if CLASSIFICATION == False:
                    outputs = self.model(**batch_inputs).squeeze()
                    loss = self.loss_function(outputs, batch_labels)
                    st_loss = self.loss_function(stupid, batch_labels)
                    test_loss.append(loss.item())
                    naive_loss.append(st_loss.item())
                    if step%512 == 0:
                        print("Labels: ", batch_labels)
                        print("Stupid: ", stupid)
                        print("Outputs: ", outputs)
                        print("------------------------")

                    metric_value = metric(outputs, batch_labels)
                    st_metric = self._r2_score(stupid, batch_labels)
                    test_metric.append(metric_value.item())
                    naive_metric.append(st_metric.item())
                else:
                    torch.set_printoptions(sci_mode=False)
                    outputs = self.model(**batch_inputs)['logits'].squeeze()
                    loss = self.loss_function(outputs, batch_labels)
                    test_loss.append(loss.item())
                    
                    naive_loss= [0]
                    naive_metric = [0.2]
                    metric_value = metric(outputs, batch_labels)
                    test_metric.append(metric_value.item())
                    if step%512 == 0:
                        print("Labels: ", batch_labels)
                        print("Preds: ", torch.argmax(outputs, dim=1))
                        print("Outputs: ", torch.softmax(outputs, 1))
                        print("Metric: ", metric_value)
                        print("------------------------")
            if step >= EVALUATE_STEPS: 
                break
        
        self.writer.add_scalar("Test loss", sum(test_loss)/len(test_loss), epoch)
        self.writer.add_scalar("Test metric", sum(test_metric)/len(test_metric), epoch)
        
        print("Mean loss[{loss}] | Mean metric[{metric}]".format(loss= sum(test_loss)/len(test_loss), metric=sum(test_metric)/len(test_metric)))
        print("Stupid loss[{loss}] | Naive soulution metric[{metric}]".format(loss= sum(naive_loss)/len(naive_loss), metric=sum(naive_metric)/len(naive_metric)))

    def _accuracy(self, outputs, labels):
        preds = torch.argmax(outputs, dim=1)
        corrects = torch.sum(preds == labels)
        return torch.sum(corrects)/preds.size(dim=0)
    
    def _r2_score(self, outputs, labels):
        labels_mean = torch.mean(labels)
        ss_tot = torch.sum((labels - labels_mean) ** 2)
        ss_res = torch.sum((labels - outputs) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2

    #SAVING AND LOADING FUNCTIONS
    #These functions are used to take checkpoints every epoch
    #----------------------------------------------------------

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
