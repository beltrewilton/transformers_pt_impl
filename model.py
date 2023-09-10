import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        """constructor method
 
        Args:
            d_model (int): Dimension of the model
            vocab_size (int): vocabulary size how many worlds in the vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """In the embedding layer we multiply those weights by sqrt of d_model

        Args:
            x (_type_): weights
        """
        self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Args:
            d_model (int): Size of vector that the positional encoding shold be
            seq_len (int): Maximun length of the sequence, cfeating one vector for
            for each position 
            dropout (float): make the model less overfit, is a technique for drop 
            "one block out" 
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        



