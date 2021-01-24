import torch.nn as nn

class Embeddings(nn.Module):

    def __init__(self, embed_size, text):
        super(Embeddings, self).__init__()
        self.embed_size = embed_size
        self.src = nn.Embedding(num_embeddings=len(text.src), embedding_dim=self.embed_size, padding_idx=text.src['<pad>'])
        self.tar = nn.Embedding(num_embeddings=len(text.tar), embedding_dim=self.embed_size, padding_idx=text.tar['<pad>'])
        