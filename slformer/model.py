import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_mean



class AttTransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, *args, **kwargs):
        super(AttTransformerEncoderLayer, self).__init__(*args, **kwargs)
        self.attention_weights = None  # To store the attention weights

    def forward(self, src, *args, **kwargs):
        # Override the forward method to capture the attention weights
        output, self.attention_weights = self.self_attn(
            src, src, src, need_weights=True, attn_mask=kwargs.get('attn_mask'))
        return super().forward(src, *args, **kwargs)


class SLformer(nn.Module):

    def __init__(self, config):
        super(SLformer, self).__init__()

        assert config['d_model'] % config['n_head'] == 0, "nheads must divide evenly into d_model"
        self.config = config
        self.d_model = config['d_model']
        self.use_cross_att = config['add_att']

        self.pos_encoder = PositionalEncoding(
            d_model=config['d_model'],
            dropout=config['dropout'],
            vocab_size=config['vocab_size'],
        )

        encoder_layer = AttTransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['n_head'],
            dim_feedforward=config['transformer_hidden_dim'],
            dropout=config['dropout'],
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['transformer_num_layers'],
        )

        if self.use_cross_att:
            self.fusion_model = CrossAttention(config['d_model'], 
                                               num_layers=config['att_num_layers'], 
                                               num_heads=config['att_nhead'], 
                                               batch_norm=False, 
                                               activation="relu")

        predictor_input_dim = config['d_model']*2
        
        self.predictor = MLP(num_layers=2, 
                             input_dim=predictor_input_dim, 
                             hidden_dim=config['mlp_hidden_dim'], 
                             output_dim=config['mlp_output_dim'])


    def encode_gsent(self, x, mask):

        mask = (1-mask).bool()
        x = x.transpose(0,1)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        h = self.transformer_encoder(x, src_key_padding_mask=mask)

        return h


    def forward(self, x1, mask1, x2, mask2):

        h1 = self.encode_gsent(x1, mask1)
        h2 = self.encode_gsent(x2, mask2)

        if self.use_cross_att:
            ## input sentences
            fused = self.fusion_model(h1, h2, mask1, mask2)
        else:
            ## directly use the head gene representation
            h1_head = h1[0,:,:]
            h2_head = h2[0,:,:]
            fused = torch.cat([h1_head, h2_head], dim=1)

        out = self.predictor(fused)
        
        return out
    

    def output_att(self, x1, mask1, x2, mask2):

        h1 = self.encode_gsent(x1, mask1)
        h2 = self.encode_gsent(x2, mask2)
        for _, layer in enumerate(self.transformer_encoder.layers):
            transformer_att.append(layer.attention_weights)
        ## [batch_size, num_layer, sent_len, sent_len]
        transformer_att = torch.stack(transformer_att, dim=1)

        ## cross att
        cross_att = self.fusion_model.output_att(h1, h2, mask1, mask2)
        return cross_att, transformer_att


    def output_emb(self, x1, mask1, x2, mask2):

        h1 = self.encode_gsent(x1, mask1)
        h2 = self.encode_gsent(x2, mask2)

        if self.use_cross_att:
            fused = self.fusion_model(h1, h2, mask1, mask2, cat=False)
            return [h1,h2],fused
        else:
            return [h1,h2]
    


class PositionalEncoding(nn.Module):
    """
    standard sinusoidal positional encoding
    reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

        

class CrossAttention(nn.Module):

    def __init__(self, hidden_dim=512, num_layers=1, num_heads=8, batch_norm=False, activation="relu"):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        if batch_norm:
            self.bn1 = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
            self.bn2 = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def forward(self, x1, x2, mask1, mask2, cat=True):

        for i, layer in enumerate(self.layers):

            if i == 0:
                x1 = x1.transpose(0, 1)
                x2 = x2.transpose(0, 1)

            x1, x2 = layer(x1, x2, mask1.bool(), mask2.bool(), output_att=False)

            if self.batch_norm:
                x1 = self.bn1[i](x1.transpose(1, 2)).transpose(1, 2)
                x2 = self.bn2[i](x2.transpose(1, 2)).transpose(1, 2)
            if self.activation:
                x1 = self.activation(x1)
                x2 = self.activation(x2)

        x1_output = scatter_mean(x1, mask1, dim=1)[:, 1:,].squeeze(1)
        x2_output = scatter_mean(x2, mask1, dim=1)[:, 1:,].squeeze(1)

        if cat:
            return torch.cat([x1_output, x2_output], dim=-1)

        return [x1_output, x2_output]
    

    def output_att(self, x1, x2, mask1, mask2):

        for i, layer in enumerate(self.layers):

            if i == 0:
                x1 = x1.transpose(0, 1)
                x2 = x2.transpose(0, 1)

            att_all_tmp = [[],[],[],[]]
            att_all = []

            ## alpha11, alpha22, alpha12, alpha21
            att_total = layer(x1, x2, mask1.bool(), mask2.bool(), output_att=True)
            for j, att in enumerate(att_total):
                att_all_tmp[j].append(att)
            
            for att_part in att_all_tmp:
                ## [l,b,n,n,a]
                att_stack = torch.stack(att_part, dim=0)
                ## [b,l*a,n,n]
                b,n = att_stack.shape[1], att_stack.shape[2]
                att_stack = att_stack.permute(1, 0, 4, 2, 3).reshape(b,-1,n,n)
                att_all.append(att_stack)

        return att_all


class CrossAttentionBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super(CrossAttentionBlock, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads))
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value2 = nn.Linear(hidden_dim, hidden_dim, bias=False)


    def _alpha_from_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def _heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def forward(self, input1, input2, mask1, mask2, output_att=False):

        query1 = self._heads(self.query1(input1), self.num_heads, self.head_size)
        key1 = self._heads(self.key1(input1), self.num_heads, self.head_size)
        query2 = self._heads(self.query2(input2), self.num_heads, self.head_size)
        key2 = self._heads(self.key2(input2), self.num_heads, self.head_size)
        logits11 = torch.einsum('blhd, bkhd->blkh', query1, key1)
        logits12 = torch.einsum('blhd, bkhd->blkh', query1, key2)
        logits21 = torch.einsum('blhd, bkhd->blkh', query2, key1)
        logits22 = torch.einsum('blhd, bkhd->blkh', query2, key2)

        alpha11 = self._alpha_from_logits(logits11, mask1, mask1)
        alpha12 = self._alpha_from_logits(logits12, mask1, mask2)
        alpha21 = self._alpha_from_logits(logits21, mask2, mask1)
        alpha22 = self._alpha_from_logits(logits22, mask2, mask2)

        value1 = self._heads(self.value1(input1), self.num_heads, self.head_size)
        value2 = self._heads(self.value2(input2), self.num_heads, self.head_size)
        output1 = (torch.einsum('blkh, bkhd->blhd', alpha11, value1).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha12, value2).flatten(-2)) / 2
        output2 = (torch.einsum('blkh, bkhd->blhd', alpha21, value1).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha22, value2).flatten(-2)) / 2
        
        if not output_att:
            return output1, output2
        
        else:
            att11 = nn.functional.softmax(alpha11, dim=2)
            att22 = nn.functional.softmax(alpha22, dim=2)
            att12 = nn.functional.softmax(alpha12, dim=2)
            att21 = nn.functional.softmax(alpha21, dim=2)
            return [att11, att22, att12, att21]

    
        
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        simple MLP with batchnorm and relu
        """
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.linears.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))

        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
    def forward(self, x):

        h = x
        for i in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[i](self.linears[i](h)))
        return self.linears[-1](h)
    

        