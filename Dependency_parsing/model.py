import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ParserModel(nn.Module):

    def __init__(self, embeddings, args):
        super(ParserModel, self).__init__()
        self.n_features = args['n_features']
        self.n_classes = ['n_classes']
        self.dropout_prob = ['dropout_prob']
        self.embed_size = embeddings.shape[1]
        self.hidden_size = ['hidden_size']
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0], self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(torch.tensor(embeddings))

        self.embed_to_hidden = nn.Linear(self.embed_size*self.n_features, self.hidden_size)
        init.xavier_uniform_(self.embed_to_hidden.weight)
        
        self.dropout = nn.Dropout(self.dropout_prob)
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes)
        init.xavier_uniform_(self.hidden_to_logits.weight)

    def embedding_lookup(self, t):
        x = self.pretrained_embeddings(t)
        x = x.view(-1, self.n_features * self.embed_size) 
        return x

    def forward(self, t):
        t = self.embedding_lookup(t)
        t = self.embed_to_hidden(t)
        t = F.relu(t)
        t = self.dropout(t)
        logits = self.hidden_to_logits(t)
        return logits
