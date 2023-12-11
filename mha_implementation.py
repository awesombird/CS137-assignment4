import torch
import math


def mha_137(query, key, value, in_wbs, out_wb):
    """ An implementation of the MHA calculation. 

        args:
            query: a tensor with shape (seq_len1, batch_size, emb_dim)
            key: a tensor with shape (seq_len2, batch_size, emb_dim)
            value: a tensor with shape (seq_len2, batch_size, emb_dim)
            in_wbs: weights and bias used in linear transformations for the three input
            out_wb: weights and bias used in the last linear transformation for computing the output 
        returns:
            output: a tensor with shape (seq_len1, batch_size, emb_dim)
    """
    
    # You are supposed to implement multihead attention in this function.  


    # TODO: Please check the documentation of MHA before you implement this function
    # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

    # TODO: please read the comments below carefully to understand the provided parameters


    # We will need the following parameters to do linear transformations  
    # `wqs` is a list of weight matrices, each for an attention head. 
    # The length of `wqs` is the number of attention heads. Each element 
    # should be applied for the linear transformation of the query `query`. 

    # The list `bqs` contains bias vectors for transoformations  
    # of the query in different attention heads. The transformation of the query should be

    # query * wqs[head].transpose() + bqs[head] 

    # Here * is the matrix multiplication. Please consider the function 
    # `torch.nn.functional.linear`
    #
    # Similarly, `(wks, bks)` is for the transformations of `key`, and `(wvs, bvs)` is 
    # for the transformation of `value`

    wqs, wks, wvs, bqs, bks, bvs = in_wbs
    out_w, out_b = out_wb

#  notes
# num_heads is k
# num of columns? is d
# H_i = softmax ( Q * W_qi * W_ki^T * K^T ) V W_vi
# H = [H_1, H_2, ... H_L] * W_o
# If H has dim d' then H_1 has dim d'/L


    # Suggestion: you may want to first transpose the tensor such that the last two ranks are [seq_len, emb_dim],  
    # which is convenient for matrix calculation later.   
    query = torch.transpose(query, 0, 1) # same as torch.permute(query, 1, 0, 2)
    key = torch.transpose(key, 0, 1)
    value = torch.transpose(value, 0, 1)

    # loop over attention heads
    output = [] 
    for head in range(len(wqs)):
    
        # TODO: run linear transformation on query with (wqs[head], bqs[head])
        q_head = torch.nn.functional.linear(query, wqs[head], bqs[head])
        # TODO: run linear transformation on key with (wks[head], bks[head])
        k_head = torch.nn.functional.linear(key, wks[head], bks[head])
        # TODO: run linear transformation on value with (wvs[head], bvs[head])
        v_head = torch.nn.functional.linear(value, wvs[head], bvs[head])

        print("Q:", q_head.shape)
        print("K:", k_head.shape)
        print("V:", v_head.shape)

        # TODO: calculate attention logits using inner product
        # Please remember to scale these logits weight the sqrt of the dimmension of the transformed queries/keys
        logit = torch.matmul(q_head, torch.transpose(k_head, -1, -2)) / math.sqrt(q_head.shape[-1])

        print(logit.shape)
        # TODO: apply softmax to compute attention weights
        weights = torch.nn.functional.softmax(logit, -1)

        # TODO: use attention weights to pool values. Note that this step can be done with matrix multiplication
        output_head = torch.matmul(weights, v_head)
    
        output.append(output_head)
    
    # TODO: concatenate attention outputs from different heads 
    attn_output = torch.cat(output, dim=-1)

    # TODO: apply the last linear transformation with (out_w, out_b)
    attn_output = torch.nn.functional.linear(attn_output, out_w, out_b)
    
    # Suggestion: if necessary, please permute ranks
    attn_output = attn_output.transpose(0, 1)

    return attn_output

