import math
import torch
from torch import Tensor

from third_party import batchify, get_batch, generate_square_subsequent_mask, PositionWiseFFN, AddNorm, evaluate

# TODO: you are supposed to implement a language model and the training function to support the notebook. This is 
# the last assignment of the class, so you should have known the pipeline of training a deep model, so this time 
# the minimum starter code is given. 

# NOTE 1: this time you ARE allowed to copy-paste some code from ONLY the following two sources. You need the 
# put the code in classes or functions in `third_party.py` and import it from there, just as the commented line 
# shown above. You should uncomment the commented importing line if you need these functions 

# * d2l Chapter 11.7 [link](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)
# * Torch tutorial on language modeling [link](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

# This code file should only contain the given code and code you will type in. NO code should copy-pasted from 
# ANY source. 

# NOTE 2: You cannot import a transformer mdoel entirely from torch. You are supposed to construct a Transformer model
# with torch layers. 


# The `PositionalEncoding` class is provided here. 
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoderLayer(torch.nn.Module):
    
    def __init__(self, vocabSize, dropout=0.2, dimHidden=100, nHead=2, mask=None, device='cpu'):
        """
        args: 
            vocabulary: a list of characters. The list also provide the vocabulary size  
        """
        super().__init__()

        # Suggestion: my suggestion is to use the embedding layer to get vector representations of 
        # integers (indices of characters). An alternative is to convert indices to one-hot encodings
        # and then apply a dense layer.

        self.vocabSize = vocabSize
        self.dropout = dropout
        self.dimHidden = dimHidden
        self.numHeads = nHead
        self.mask = mask
        self.device = device

        # Sub-modules
        self.multiHeadAttn = torch.nn.MultiheadAttention(self.vocabSize, self.numHeads, dropout=0)
        self.addNorm1 = AddNorm(self.vocabSize, dropout=self.dropout)
        self.ffn = PositionWiseFFN(self.dimHidden, self.vocabSize)
        self.addNorm2 = AddNorm(self.vocabSize, dropout=self.dropout)

    def forward(self, X):
        # You will need a mask for causal modeling, and `generate_square_subsequent_mask` 
        # will get you the mask

        mask = generate_square_subsequent_mask(X.size(0))
        mask = mask.to(self.device)

        # attn_output_weights - Only returned when need_weights=True
        attn_output, attn_output_weights = self.multiHeadAttn(X, X, X, attn_mask=mask, is_causal=True)
        attn_output = attn_output.to(self.device)

        norm = self.addNorm1(X, attn_output)
        ffn = self.ffn(norm)
        norm2 = self.addNorm2(ffn, norm)

        return norm2



class SmallLanguageModel(torch.nn.Module):
    """
    A small language model using the transformer architecture
    """

    def __init__(self, vocabulary, dropout=0.2, dimHidden=100, nlayers=2,  nHead=2, device='cpu'):
        """
        args: 
            vocabulary: a list of characters. The list also provide the vocabulary size  
        """
        super().__init__()

        # Suggestion: my suggestion is to use the embedding layer to get vector representations of 
        # integers (indices of characters). An alternative is to convert indices to one-hot encodings
        # and then apply a dense layer.

        self.vocabulary = vocabulary
        self.vocabSize = len(self.vocabulary)
        self.numHeads = nHead
        self.numlayers = nlayers
        self.device = device
        self.dropout = dropout
        self.dimHidden = dimHidden

        # print(self.vocabSize, self.numHeads, self.vocabSize/self.numHeads)
        # Sub-modules
        self.inputEmb = torch.nn.Embedding(num_embeddings=self.vocabSize, embedding_dim=self.vocabSize)
        self.posEnc = PositionalEncoding(self.vocabSize)
        self.linear = torch.nn.Linear(self.vocabSize, self.vocabSize)
        self.transLayers = torch.nn.ModuleList([TransformerEncoderLayer(self.vocabSize, dropout=self.dropout, dimHidden=self.dimHidden, nHead=self.numHeads, device=self.device)
                                                for _ in range(self.numlayers)])
        self.layers = []

    def forward(self, X):
        """
        The forward function of the model. We will follow the format of the `torch.nn.Transformer` and assume 
        `X` has the shape `[seq_len, batch_size]`. 
        args:
            X: a tensor with shape `[seq_len, batch_size]`. Each entry of X is an index to a char in vocabulary. 
        returns:
            out: a tensor with shape `[seq_len, batch_size, len(vocabulary)]`. The fiber `X[t, b, :]` should be the logits 
                 for the prediction of the `(t+1)`-th char in the input sequence. 
        """

        # print("X: ", X.size(0), X.size(1))
        # mask = generate_square_subsequent_mask(X.size(1))
        X = X.to(self.device)
        inputs = self.inputEmb(X)
        inputs = self.posEnc(inputs)

        output = inputs
        for layer in self.transLayers:
            output = layer(output)

        return output

def train_helper(model, train_loader, loss_func, optimizer, bptt=50, device="cpu"):
    
    running_loss = 0.0
    log_interval = 2000
    log_count = 0
    
    for i in range(0, train_loader.size(0), bptt):
        data, targets = get_batch(train_loader, i, bptt)
        data = data.long()
        targets = targets.long()
        
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        
        output = model(data)
        output_flat = output.view(-1, model.vocabSize)
        loss = loss_func(output_flat, targets)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        
        if i / bptt >= log_interval * log_count:
            # running_loss / ((i / bptt) - log_interval * log_count + 1)
            print("Index:", i, "Loss:", loss.item() / bptt)
            log_count += 1
            running_loss = 0.0
        
        
    


def train(model, train_data, val_data, loss_func, optimizer, scheduler, num_epochs = 2, bptt = 50, batch_size=32, device="cpu"):
    """
    The training function for language modeling

    args: 
        model: a language model. Given (c_0, c_2, ..., c_{k-1}), then model should output logits for (c_1, c_2, ..., c_k) 
        train_data: a 1-d tensor containing the training data
        val_data: a 1-d tensor containing the validation data
        loss_func: the loss function
        optimizer: the torch opimizer 
        schedular: the torch schedular for learning rate 
        num_epochs: int, the maximum number of training epochs 
        bptt: int, the window size of the input, or it is the sequence length in one batch.    
    """
     
    model = model.to(device)
    
    # Track only the best model
    best_val_loss = float("inf")
    best_model = None
    
    train_loader = batchify(train_data, batch_size)
    val_loader = batchify(val_data, batch_size)
    val_loader = val_loader.to(device)

    for epoch in range(num_epochs):
        model.train()
        # start_time = time.time()
        
        train_helper(model, train_loader, loss_func, optimizer, bptt, device)
        
            
        val_loss = evaluate(model, val_loader, loss_func, bptt)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
        
        # Update the scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        print(f'End of epoch {epoch} | Validation loss: {val_loss:.2f}')

    model.load_state_dict(best_model.state_dict())
