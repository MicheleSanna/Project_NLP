import time 
import torch

def check_parameters_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            print("Freezed: ", name)

def print_potato(worker_id):
    print("Potato: ", worker_id)

def take_time_of_a_function(fun, args):
    start_time = time.time()
    ret = fun(**args)
    print("Tempo: ", (time.time()-start_time))
    return ret

def bert_test(model, tokenizer):
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    print(encoded_input)
    output = model(**encoded_input)
    print(output['logits'])
    print(output.__dict__.keys())
    #english_text = tokenizer.batch_decode(output)
    #print(output['last_hidden_state'].size())
    #print(output['hidden_states'].size())
    #print(output['attentions'].size())
    #print(output['pooler_output'].size())

def gpt2_test(model, tokenizer):
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    print(encoded_input)
    output = model(**encoded_input)
    loss, logits = output[:2]
    print(loss)
    print(logits)


def test_dataloader( dataloader, full = False):
    if full:
        for step, batch in enumerate(dataloader):
            print("BATCH: ", batch)
            for labels, elements in batch:
                print("NO SQUEEZE: ", labels)
                print("NO SQUEEZE: ",elements)
                for value in elements.values():
                    print("SIZE: ", value.size())
                    #print("SIZE WITH SQUEEZE: ", torch.squeeze(value, dim=1).size())
                    value = torch.squeeze(value, dim=1)
                    print("SIZE SQUEEZE: ", value.size())

                elements['input_ids'] = torch.squeeze(elements['input_ids'], dim=1)
                elements['token_type_ids'] = torch.squeeze(elements['token_type_ids'], dim=1)
                elements['attention_mask'] = torch.squeeze(elements['attention_mask'], dim=1)
            break
    else:
        for step, batch in enumerate(dataloader):
            print("BATCH: ", batch)
            break

def compare_model_param(model1, model2, device):
    model1.to(device)
    model2.to(device)
    for param1, param2 in zip(model1.named_parameters(), model2.named_parameters()):
        name1, val1 = param1
        name2, val2 = param2
        if (name1 == name2):
            print("Distance of param{name}: {value}".format(name= name1, value=torch.norm(val1-val2)))

def predict(tokenizer, model, text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)['logits']
    for i in range(5):
        print("Probabilità che la recensione sia da {n} stelle: {x}%".format(n= i, x= output[i]*100))
    