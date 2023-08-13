'''
Author: Zhang
Date: 2022-08-10
LastEditTime: 2023-01-20
FilePath: /code_attack/csa/model/GraphCodeBert.py
Description: 

'''
import torch
from torch import nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer,RobertaConfig

       
class GraphCodeBERTClassifier(nn.Module):   
    def __init__(self, **config):
       
        super(GraphCodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(config['model_name'], num_labels=config['num_classes'])
        self.embed = self.model.roberta.embeddings.word_embeddings# embedding,Embedding(batch, 768, padding_idx=1)
        self.config=config
        self.vocab_size=config['vocab_size']
        self.embedding_size = self.embed.weight.size()[-1]
        
        
    def forward(self, inputs_ids,position_idx,attn_mask,labels=None): 
      
        
        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)    
        inputs_embeddings=self.model.roberta.embeddings.word_embeddings(inputs_ids)
        #nodes_mask[:,:,None]=>[batch,320,1],token_mask[:,None,:]=>[batch,1,320],attn_mask:batch,320,320] 
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]   
       
        # outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx,token_type_ids=position_idx.eq(-1).long()).last_hidden_state# 
        outputs=self.model(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx,labels=labels)
        # shape: [batch_size, num_classes]
        # prob=F.softmax(logits, dim=-1)
        if labels is not None:
            loss, logits = outputs.loss, outputs.logits
            return logits, loss
        else:
            return outputs.logits
    def prob(self, inputs_ids,position_idx,attn_mask,labels):
        
        with torch.no_grad():
            logits, loss = self.forward(inputs_ids,position_idx,attn_mask,labels)
            prob = nn.Softmax(dim=-1)(logits)
        return prob,loss

    def grad(self, inputs_ids,position_idx,attn_mask,labels):
       
        
        self.zero_grad()
        self.embed.weight.retain_grad() 
        logits, loss = self.forward(inputs_ids,position_idx,attn_mask,labels)
        loss.backward()
        return self.embed.weight.grad