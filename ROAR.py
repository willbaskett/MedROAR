import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from tqdm.notebook import tqdm
from ROPE import RotaryEmbedding

debug = False
torch._dynamo.config.cache_size_limit = 64
#torch._dynamo.config.guard_nn_modules = True

#make alphas for B, L sequences
def make_non_uniform_alphas1D(x, smoothing_ratio=0, return_probability_dist=True):
    assert len(x.shape) == 2
    device = x.device
    B, L = x.shape
    pool_width = int(L*smoothing_ratio)*2+1
    step_multiplier = 1.1
    noise_magnitude = 1
    normal_dist = torch.distributions.Normal(0,1)
    base = torch.zeros((B, 1, L), device=device)
    dimensions = L
    while int(dimensions) > 0:
        rand = torch.empty((B, 1, int(dimensions)), device=device).normal_(0,1) * noise_magnitude
        base += F.interpolate(rand, L, mode="linear")
        dimensions = dimensions/step_multiplier
        noise_magnitude *= step_multiplier
    
    base = base.reshape(B, L)
    base = F.avg_pool1d(base, pool_width,1, padding=pool_width//2, count_include_pad=False)
    base = base - base.mean(axis=1).unsqueeze(1)
    base = base / base.std(axis=1).unsqueeze(1)
    
    if return_probability_dist:
        dist = normal_dist.cdf(base)
        dist = dist - dist.min(axis=1)[0].unsqueeze(1)
        dist = dist / dist.max(axis=1)[0].unsqueeze(1)
        return dist
    else:
        return base

#scrambles the order of input data, accepts lists of tabular and sequence data
def scramble_order(tab_data=[], seq_data=[], device=None, biased_scramble=True, smoothing_ratio=0, return_index=False, simple_scramble=False):
    assert device is not None
    if not isinstance(tab_data, list):
        tab_data = [tab_data]
    if not isinstance(seq_data, list):
        seq_data = [seq_data]
    assert len(tab_data) + len(seq_data) > 0
        
    length = 0
    batch_size = None
    tab_data_list = []
    for t in tab_data:
        x_type_tab, x_tab, pos_tab = t
        x_tab, x_type_tab, pos_tab, = x_tab.to(device), x_type_tab.to(device), pos_tab.to(device)
        tab_data_list.append((x_tab, x_type_tab, pos_tab))
        length += x_tab.shape[1]
        if batch_size is None:
            batch_size = x_tab.shape[0]

    seq_data_list = []
    for s in seq_data:
        x_type_seq, x_seq, pos_seq, = s
        x_seq, x_type_seq, pos_seq, = x_seq.to(device), x_type_seq.to(device), pos_seq.to(device),
        seq_data_list.append((x_seq, x_type_seq, pos_seq))
        length += x_seq.shape[1]
        if batch_size is None:
            batch_size = x_seq.shape[0]
            
    alphas = torch.empty((batch_size, length), device=device).uniform_(0, 0.000001)

    if biased_scramble:
        tab_alphas = []
        for t in tab_data_list:
            x_tab, x_type_tab, pos_tab = t
            L = x_tab.shape[1]
            if simple_scramble:
                tab_alphas.append(torch.randint(0,2, (batch_size,1), device=device).float().repeat(1,L))
            else:
                tab_alphas.append(torch.empty((batch_size,1), device=device).uniform_(0,1).repeat(1,L))
    
        seq_alphas = []
        for s in seq_data_list:
            x_seq, x_type_seq, pos_seq = s
            L = x_seq.shape[1]
            if simple_scramble:
                alphas_seq = torch.arange(L, device=device)
                alphas_seq = alphas_seq/alphas_seq.max()
                alphas_seq = alphas_seq.repeat(batch_size, 1).reshape(batch_size,-1)
                seq_alphas.append(alphas_seq)
            else:
                alphas_seq = make_non_uniform_alphas1D(x_seq, smoothing_ratio=smoothing_ratio)
                seq_alphas.append(alphas_seq)

        
        extra_alphas = torch.cat(tab_alphas+seq_alphas, axis=1)
        alphas += extra_alphas

    x_val = [x[0] for x in tab_data_list] + [x[0] for x in seq_data_list]
    x_type = [x[1] for x in tab_data_list] + [x[1] for x in seq_data_list]
    x_pos = [x[2] for x in tab_data_list] + [x[2] for x in seq_data_list]

    data_type = torch.cat(x_type, axis=1)
    data_val = torch.cat(x_val, axis=1)
    data_pos = torch.cat(x_pos, axis=1)
    
    _, index = alphas.sort(axis=1, descending=False)

    x_type_reordered = torch.gather(data_type, 1, index)
    x_reordered = torch.gather(data_val, 1, index)
    seq_order = torch.gather(data_pos, 1, index)
    
    if return_index:
        return index, x_type_reordered, x_reordered,  (seq_order,)
    else:
        return x_type_reordered, x_reordered, (seq_order,)

def make_diag(length):
    array = torch.ones((length+1, length))
    for i in range(0,length+1):
        array[i,i:] = 0
    return array

class ToBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = torch.round(x)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone() # pass-through

class BinaryQuantizer(nn.Module):
    def forward(self, x, add_noise=False, round_during_training=True):
        if add_noise and self.training:
            x = x + torch.empty_like(x).normal_()
        x = torch.sigmoid(x)
        if not self.training or round_during_training: 
            x = ToBinary.apply(x)
        return x

####################################################################################
# Transformer Code Autoencoder
####################################################################################

#Applies a binary mask to a vector (the embedding) then does things to allow the network to gracefully handle masked items and distinguish them from actual 0s
#For an N length embedding vector:
#1) Multiply by the binary mask, zeroing out masked item
#2) Concat the mask and embedding vector to make a 2 X N matrix
#3) Multiply each element in this matrix by a learned weight and add a learned bias term
#4) Pass each Mask/Embedding element pair through a neural network (same network for all items, applied separately)
class ApplyMask(nn.Module):
    def __init__(self, num_bits, is_binary):
        super(ApplyMask, self).__init__()
        self.is_binary = is_binary
        self.element_weight = nn.Parameter(torch.ones((num_bits, 2)))
        self.element_bias = nn.Parameter(torch.zeros((num_bits, 2)))
        self.fc1 = nn.Linear(2, 1024)
        self.fc2 = nn.Linear(512, 1)
        self.swiglu = SwiGLU()

    def forward(self, x, mask):
        in_shape = x.shape
        if self.is_binary:
            x = x*2 - 1
        x = x * mask
        x = torch.stack([x,mask], 2)
        x = x * self.element_weight + self.element_bias
        return self.fc2(self.swiglu(self.fc1(x))).reshape(in_shape)

class ProbDropout(nn.Module):
    def __init__(self, min_p=0, max_p=1):
        super(ProbDropout, self).__init__()
        self.min_p = min_p
        self.max_p = max_p
        
    def forward(self, x):
        if self.training:
            mask = torch.empty_like(x).uniform_(0,1) > torch.empty((x.shape[0],1,1), device=x.device).uniform_(self.min_p, self.max_p)
            x = x * mask
        return x

#Implements Swish gated linear units
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

#Implements the positionwise feedforward network for the transformer
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff*2)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.swiglu = SwiGLU()

    def forward(self, x):
        return self.fc2(self.swiglu(self.fc1(x)))

#Implements a basic feedforward network applied to translate transformer outputs into the enbedding (encoder) and from the embedding back to the input to a transformer (decoder)
class TranslationFF(nn.Module):
    def __init__(self, d_reserved, d_model, d_ff):
        super(TranslationFF, self).__init__()
        self.norm = nn.LayerNorm(d_reserved+d_model)
        self.fc1 = nn.Linear(d_reserved+d_model, d_ff*2)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.swiglu = SwiGLU()
    
    def forward(self, reserved, x):
        x_base = x.clone()
        x = torch.cat([reserved, x], axis=1)
        return self.fc2(self.swiglu(self.fc1(self.norm(x)))) + x_base

#Implements a layer of the transformer encoder with rotary positional encodings
#Can rotate keys/queries across multiple axis if input is more than 1d (like an image)
#A portion of each k/v is left unrotated, how much depends on how many axis need to be encoded
class EncoderLayer(nn.Module):
    def __init__(self, n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=1, dropout_max_p=0):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = num_heads
        self.d_ff = d_ff
        self.num_rotation_axis = num_rotation_axis
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.type_embedding = nn.Embedding(n_type_embedding, 2 * d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dim_per_rope = (d_model//num_heads)//(num_rotation_axis+1)
        self.rotary_emb = RotaryEmbedding(dim = self.dim_per_rope, cache_if_possible=False)
        self.dropout = ProbDropout(max_p=dropout_max_p)
        

    #takes type, value, position, triplets
    #each is a vector of the same length where each element describes type, value, or position about the input data
    #seq_order is a tuple which can contain multiple axis
    def forward(self, x_type, x_value, seq_order):
        B, T, C = x_value.size()

        #eake queries, keys, values
        q, k, v  = self.c_attn(self.norm1(x_value)).split(self.d_model, dim=2)

        #encodes the TYPE of value into the key/queries by adding a linear projection to each k/q
        q_embed, k_embed = self.type_embedding(x_type).split(self.d_model, dim=2)
        q = q + q_embed
        k = k + k_embed

        #break k/q/v into distinct heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        #Add rotations to the correct portion of the keys/queries using the positions listed in the seq_order tuple
        for i, s in enumerate(seq_order):
            q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s.unsqueeze(1))
            k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s.unsqueeze(1))

        #Use flash attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.dropout(attn_output)
        
        x_value = x_value + attn_output
        
        ff_output = self.feed_forward(self.norm2(x_value))
        ff_output = self.dropout(ff_output)
        
        x_value = x_value + ff_output
        return x_value

#Implements a tranformer encoder which produces a single output vector
#Options allow it to encode into binary vectors and to create embeddings based on partial inputs
class Encoder(nn.Module):
    def __init__(self, n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=False, learn_partial_encodings=False, num_rotation_axis=1, dropout_max_p=0):
        super(Encoder, self).__init__()
        self.binary_encodings = binary_encodings
        self.n_bits = n_bits
        self.learn_partial_encodings = learn_partial_encodings
        self.embedding = nn.Embedding(n_embedding, d_model)
        self.type_embedding = nn.Embedding(n_type_embedding, d_model)
        encoder = []
        for i in range(depth):
            encoder.append(EncoderLayer(n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=num_rotation_axis, dropout_max_p=dropout_max_p))
        self.encoder = nn.ModuleList(encoder)
        translation_layers = []
        for _ in range(n_translation_layers):
            translation_layers.append(TranslationFF(d_model, d_model, d_ff))
        self.translation = nn.ModuleList(translation_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.lin_out = nn.Linear(d_model, self.n_bits)
        self.binary_quantize = BinaryQuantizer()
        
    def forward(self, x_type, x_value, seq_order):
        if debug:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B, L = x_type.shape
        mask = torch.ones(L, device=device).repeat(B,1)
        if self.training and self.learn_partial_encodings:
            mask = torch.arange(L, device=device).repeat(B,1)/L < torch.empty((B,1), device=device).uniform_(0,1)
        
        #add pooling token pos
        new_seq_order = []
        for s in seq_order:
            new_seq_order.append(torch.cat([torch.zeros((B,1), device=device),s*mask], axis=1).float())
        seq_order = new_seq_order

        #add pooling token, pooling is done at this token. If strings include CLS the CLS will be scrambled like all other tokens and just marks the start
        x_value = torch.cat([torch.ones((B,1), device=device),x_value*mask], axis=1).long()
        x_type = torch.cat([torch.ones((B,1), device=device),x_type*mask], axis=1).long()

        #initial input to the transformer
        x_value = self.embedding(x_value) + self.type_embedding(x_type)
        
        for block in self.encoder:
            x_value = block(x_type, x_value, seq_order=seq_order)
        x_value = x_value[:,0,:]
        base_value = x_value.clone()
        for block in self.translation:
            x_value = block(base_value, x_value)
        x_value = self.lin_out(self.norm_out(x_value))

        if self.binary_encodings:
            x_value = self.binary_quantize(x_value)
        elif self.training: 
            x_value = x_value + torch.empty_like(x_value).normal_(std=0.01)
            
        return x_value

#Implements a layer of the transformer decoder with rotary positional encodings
#Can rotate keys/queries across multiple axis if input is more than 1d (like an image)
#A portion of each k/v is left unrotated, how much depends on how many axis need to be encoded
class DecoderLayer(nn.Module):
    def __init__(self, n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=1, dropout_max_p=0):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = num_heads
        self.d_ff = d_ff
        self.num_rotation_axis = num_rotation_axis
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.type_embedding = nn.Embedding(n_type_embedding, 2 * d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dim_per_rope = (d_model//num_heads)//(num_rotation_axis+1)
        self.rotary_emb = RotaryEmbedding(dim = self.dim_per_rope, cache_if_possible=False)
        self.dropout = ProbDropout(max_p=dropout_max_p)
    
    #CURRENT position/type is encoded into the queries while "NEXT" position is encoded into the keys
    #This allows information to be routed to allow for autoregressive prediction of "NEXT" tokens
    #Predicting "NEXT" tokens of unknown position and type is impossible so we have to give the self attenton mechanism this information
    def forward(self, x_type, x_value, seq_order):
        B, T, C = x_value.size()
        q, k, v  = self.c_attn(self.norm1(x_value)).split(self.d_model, dim=2)
        q_embed, k_embed = self.type_embedding(x_type).split(self.d_model, dim=2)

        #Encodings of type are offset for queries/keys in the decoder. See above note.
        k = k + k_embed[:,:-1]
        q = q + q_embed[:,1:]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        
        #Encodings of position are offset for queries/keys in the decoder. See above note.
        for i, s in enumerate(seq_order):
            k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s[:,:-1].unsqueeze(1))
            q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s[:,1:].unsqueeze(1))

        #Flash attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.dropout(attn_output)
        
        x_value = x_value + attn_output
        ff_output = self.feed_forward(self.norm2(x_value))
        ff_output = self.dropout(ff_output)
        x_value = x_value + ff_output
        return x_value

#Implements a tranformer decoder which takes a single vector embedding and autoregressively decodes it in a specified order
#Options allow it to use binary vectors and to create hierarchical embeddings by masking parts of the input embeddings
class Decoder(nn.Module):
    def __init__(self, n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=False, ordered_encodings=False, num_rotation_axis=1, dropout_max_p=0):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.ordered_encodings = ordered_encodings
        self.binary_encodings = binary_encodings
        self.n_bits = n_bits
        self.apply_mask = ApplyMask(n_bits, is_binary=binary_encodings)
        self.embedding = nn.Embedding(n_embedding, d_model)
        self.type_embedding = nn.Embedding(n_type_embedding, d_model)
        decoder = []
        for i in range(depth):
            decoder.append(DecoderLayer(n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=num_rotation_axis, dropout_max_p=dropout_max_p))
        self.decoder = nn.ModuleList(decoder)
        self.lin_in = nn.Linear(self.n_bits, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.lin_out = nn.Linear(d_model, n_embedding)
        
        translation_layers = []
        for _ in range(n_translation_layers):
            translation_layers.append(TranslationFF(d_model,d_model, d_ff))
        self.translation = nn.ModuleList(translation_layers)
        self.register_buffer('mask', make_diag(self.n_bits))

    #makes a mask used to force the encoder to push more information into earlier dimensions of the embedding vector
    #masks the last X items in the vector where X is random
    #this forces more information into earlier dimensions
    def make_rand_mask(self, x_in, num_allowed_nodes=None):
        if debug:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rand_mask = None
        B = x_in.shape[0]
        if self.ordered_encodings:
            #mask nodes to force more information into fewer nodes
            if num_allowed_nodes is not None:
                rand_mask = self.mask[num_allowed_nodes].repeat(B,1)
            else:
                r = torch.empty((B,), device=device).uniform_(0,1) ** 2
                r = r - r.min()
                r = r / r.max()
                r = torch.nan_to_num(r)
                rand_index = (r * (self.mask.shape[0]-1)).round().int()
                
                if self.training:
                    rand_mask = self.mask[rand_index]
                else:
                    rand_mask = self.mask[-1].repeat(B,1)
        else:
            rand_mask = self.mask[-1].repeat(B,1)
            
        return rand_mask

    def forward(self, x_type, x_value, seq_order, enc=None, num_allowed_nodes=None):
        if debug:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #makes the mask which is randomly applied to the input embedding to force a hierarchy
        #masks the last X items in the vector where X is random
        #this forces more information into earlier dimensions
        if enc is None:
            enc = torch.zeros((x_value.shape[0],self.n_bits), device=device)
            enc_mask = torch.zeros(enc.shape, device=device)
        else:
            enc_mask = self.make_rand_mask(x_value, num_allowed_nodes=num_allowed_nodes)

        #add a leading zero to the vectors containing the positions of each element due to the right shift of decoder inputs needed for autoregressive causal masked training
        new_seq_order = []
        for s in seq_order:
            new_seq_order.append(torch.cat([torch.zeros((x_value.shape[0],1), device=device),s], axis=1).float())
        seq_order = new_seq_order 

        #apply translate from embedding to a middle form before giving the vector to the transformer decoder
        enc = self.apply_mask(enc, enc_mask)
        enc = self.norm_in(self.lin_in(enc))
        base_value = enc.clone()
        
        for block in self.translation:
            enc = block(base_value, enc)
        
        enc = enc.unsqueeze(1)
        
        seq_len = x_value.shape[1]

        #add a leading one to the vectors containing the positions of each element due to the right shift of decoder inputs needed for autoregressive causal masked training
        #the token "1" identifies that this is the input token
        x_type = torch.cat([torch.ones((x_type.shape[0],1), device=device),x_type], axis=1).long()
        x_value = self.embedding(x_value) + self.type_embedding(x_type[:,:-1])

        #add the embedding to the input at the first position. Shift the sequence right one element for autoregressive training
        x_value = torch.cat([enc, x_value], axis=1)[:,0:seq_len,:]
        for block in self.decoder:
            x_value = block(x_type, x_value, seq_order=seq_order)

        x_value = self.lin_out(self.norm_out(x_value)).permute(0,2,1)
        return x_value

#Base Random Order Autoregressive Transformer Autoencoder class
#n_embedding = number of distinct VALUES for inputs
#n_type_embedding = number of distinct TYPES of inputs
#d_model base model dimension
#num_heads = number of attention heads, heads are of size d_modes//num_heads
#d_ff = dimensionality of the feedforward dimension of the feedforward layer, should be larger than d_model 2X or 4X as large are common
#depth = number of transformer layers in the encoder and decoder respectively. i.e. 8 = 8 encoder layers + 8 decoder layers
#n_translation_layers = number of feedforward layers after the encoder to translate output -> embedding and before the decoder to translate embedding -> decoder input
#n_bits = dimensionality of the embedding
#binary_encodings = round embedding values to binary. T/F
#ordered_encodings = learn hierarchical embedding vector
#make_encodings = learn to make embeddings T/F, this model works fine as a decoder only model, it doesn't have to make embeddings
#learn_partial_encodings = learn to make embeddings from incomplete input
#num_rotation_axis = number of positional embedding axis needed for the input. i.e tabular data = 0, sequence = 1, image = 2, etc.
class MedROAR(nn.Module):
    def __init__(self, n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits=256, binary_encodings=False, ordered_encodings=False, 
                 make_encodings=True, learn_partial_encodings=False, num_rotation_axis=1, dropout_max_p=0):
        super(MedROAR, self).__init__()
        self.n_bits = n_bits
        self.make_encodings = make_encodings
        self.ordered_encodings = ordered_encodings
        self.binary_encodings = binary_encodings

        if make_encodings:
            self.encoder = Encoder(n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=binary_encodings, learn_partial_encodings=learn_partial_encodings, num_rotation_axis=num_rotation_axis)
        
        self.decoder = Decoder(n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=binary_encodings, ordered_encodings=ordered_encodings, num_rotation_axis=num_rotation_axis, dropout_max_p=dropout_max_p)
        
        self.register_buffer('mask', make_diag(self.n_bits))
        self.best_complexity = -1
        
    def forward(self, decoder_input, encoder_input=None, enc=None, num_allowed_nodes=None):
        if encoder_input is None:
            encoder_input = decoder_input
        encoder_type, encoder_value, encoder_seq_order = encoder_input
        
        if enc is None and self.make_encodings:
            enc = self.encoder(encoder_type, encoder_value, encoder_seq_order)

        decoder_type, decoder_value, decoder_seq_order = decoder_input
        out = self.decoder(decoder_type, decoder_value, decoder_seq_order, enc, num_allowed_nodes=num_allowed_nodes)
        

        return enc, out
