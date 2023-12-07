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
    cosine_sim = torch.nn.CosineSimilarity(dim=2)
    text = "The pen is on the table"
    encoded_input = tokenizer(text, return_tensors='pt')
    print(encoded_input)
    output = model(**encoded_input)
    print(output.keys())
    #english_text = tokenizer.batch_decode(output)
    #print(output['last_hidden_state'].size())
    print("Pooler output size: ", output['pooler_output'].size())
    for i in range(len(output['hidden_states'])):
        print("Size of hidden state {i}: {size}".format(i=i, size=output['hidden_states'][i].size()))
    print("Last hidden state size: ", output['last_hidden_state'].size())
    #print(output['attentions'].size())
    #print(output['pooler_output'].size())
    for i in range(len(output['hidden_states'])):
        print("Similarity of hidden state {i}: {sim}".format(i=i, sim=cosine_sim(output['hidden_states'][i], output['last_hidden_state'])))

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

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def output_info(outputs):
    print("keys:", outputs.keys())
    print("Lenghth of output list: ", len(outputs))
    print("Type of element 0: ", type(outputs[0]))
    print("Size of element 0: ", (outputs[0]).size())
    print("Type of element 1: ", type(outputs[1]))
    print("Size of element 1: ", (outputs[1]).size())
    print("Type of element 2: ", type(outputs[2]))
    print("Size of element 2: ", len(outputs[2]))
    print("Type of element 3: ", type(outputs[3]))
    print("Size of element 3: ", len(outputs[3]))
    print("Type of hidden states: ", type(outputs.hidden_states))
    print("Size of hidden states: ", len(outputs.hidden_states))
    print("Type of an element of hidden states: ", type(outputs.hidden_states[1]))
    print("Size of an element of hidden states: ", outputs.hidden_states[1].size())
    ppl = torch.exp(outputs.loss)
    print("Perplexity: ", ppl)
    