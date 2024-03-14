# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sn
# import numpy as np
# import random
import torch
import torch.nn as nn
import math
# from sklearn.metrics import f1_score, precision_score, recall_score, ConfusionMatrixDisplay
# import time
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
from pytorch_pretrained_vit import ViT

config = {
    "patch_size": 16,
    "hidden_size": 768,
    "num_hidden_layers": 12,  #Number of times to repeat encoder block
    "num_attention_heads": 4,
    "intermediate_size": 4 * 768, # 4 * hidden_size
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": 128,
    "num_classes": 4, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


    
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.
    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        #In above nn.Linear hidden_size is in_features in prior implementation(Transfer Learning) and attention_head_size is out_features in the same
        #In above command Linear means full connected without activation function but it has learnable weights and it is feed forward
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        # print('shape of matmul of Q & K', attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        # self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.attention_head_size = self.hidden_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
            #The above line appends the heads to the list named self.heads
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=True):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        stack_prob = torch.stack([attention_prob for _, attention_prob in attention_outputs])
        ref_prob = torch.zeros(stack_prob[0].shape)
        ref_prob = ref_prob.to(device)
        diff = torch.tensor([torch.sum(torch.abs(stack_prob[i]-ref_prob)) for i in range(stack_prob.shape[0])])
        _, best_prob_indx = torch.sort(diff, dim = 0, descending=True)
        best_prob_indx = best_prob_indx.to(device)
        # print('The best attn prob', stack_prob[best_prob_indx[0]].shape)
        attention_best, _ = attention_outputs[best_prob_indx[0]]
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)

        #Here dim=-1 means ignoring the dimension
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        attention_output = attention_output + attention_best
        return attention_output
        

class MyFC(nn.Module):
      def __init__(self, numClass):
          super(MyFC,self).__init__()
          self.clf=nn.Sequential(
              nn.Flatten(),
              nn.Linear(768,256),
              nn.ReLU(),
            #   nn.Linear(256,128),
            #   nn.ReLU(),
              nn.Dropout(0.4),
              nn.Linear(256,numClass)
          )
      def forward(self,x):
          return self.clf(x)    
      
#%% Return pretrain ViT wit 12 encoder block trained on imageNet1k

def ViTpretrain(img_size, numClass):
    model = ViT('B_16_imagenet1k', image_size = img_size, num_classes = numClass, pretrained=True)
    model.fc = MyFC(numClass)
    return model
      
#%%  Models: EfficientNet, ResNetXt, Dense, VGGNet, GoogleNet

class EffNetFC(nn.Module):
    def __init__(self, numClass):
        super(EffNetFC,self).__init__()
        self.clf=nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280,numClass)
        )
    def forward(self,x):
        return self.clf(x)
    
    

class ResNetFC(nn.Module):
    def __init__(self, numClass):
        super(ResNetFC,self).__init__()
        self.clf=nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048,numClass)
        )
    def forward(self,x):
        return self.clf(x)
    

def ResNetXt(numClass):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    model.fc = ResNetFC(numClass)
    return model


class densNetFC(nn.Module):
    def __init__(self, numClass):
        super(densNetFC,self).__init__()
        self.clf=nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024,numClass)
        )
    def forward(self,x):
        return self.clf(x)
    

def DenseNet(numClass):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    model.classifier = densNetFC(numClass)
    return model

class vggNetFC(nn.Module):
    def __init__(self, numClass):
        super(vggNetFC,self).__init__()
        self.clf=nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088,numClass)
        )
    def forward(self,x):
        return self.clf(x)
    

def vgg16Net(numClass):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    model.classifier = vggNetFC(numClass)
    return model

class gNetFC(nn.Module):
    def __init__(self, numClass):
        super(gNetFC,self).__init__()
        self.clf=nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024,numClass)
        )
    def forward(self,x):
        return self.clf(x)
    

def googleNet(numClass):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    model.fc = gNetFC(numClass)
    return model


class alexNetFC(torch.nn.Module):
    def __init__(self, numClass):
        super(alexNetFC, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512,numClass)
        )

    def forward(self, x):
        return self.classifier(x)
    
def myAlexNet(numClass):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True) #Pretained model from the pytorch repo
    model.fc = alexNetFC(numClass)
    return model


class sqNetFC(nn.Module):
    def __init__(self, numClass):
        super(sqNetFC,self).__init__()
        self.clf=nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024,numClass)
        )
    def forward(self,x):
        return self.clf(x)
    

def SqeezeNet(numClass):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
    model.fc = sqNetFC(numClass)
    return model