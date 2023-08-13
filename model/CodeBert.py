'''
Author: Zhang
Date: 2022-08-10
LastEditTime: 2023-01-20
FilePath: /code_attack/csa/model/CodeBert.py
Description: pretrain model codebert

'''
import torch
from torch import nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer
class CodeBERTClassifier(nn.Module):
    
    def __init__(self,**config):
      
        
        super(CodeBERTClassifier, self).__init__()
        
        
       
        self.model = RobertaForSequenceClassification.from_pretrained(config['model_name'], num_labels=config['num_classes'])
        self.input_len = config['input_len'] 
        self.embed = self.model.roberta.embeddings.word_embeddings 
        self.vocab_size = self.embed.weight.size()[0]
        self.embedding_size = self.embed.weight.size()[-1]
        self.device = config['device']
        self.vocab=config['vocab']
    

    def forward(self, inputs, labels=None):
        
        outputs = self.model(inputs, attention_mask=inputs.ne(self.vocab.tokenizer.pad_token_id), labels=labels)
        if labels is not None:
            loss, logits = outputs.loss, outputs.logits
            return logits, loss
        else:
            return outputs.logits

    def prob(self, inputs):
        with torch.no_grad():
            logits = self.forward(inputs)
            prob = nn.Softmax(dim=-1)(logits)
        return prob

    def grad(self, inputs, labels):
       
        
        self.zero_grad()
        
        self.embed.weight.retain_grad() # (50265, 768)
        

        logits, loss = self.forward(inputs, labels)
        loss.backward()

        return self.embed.weight.grad