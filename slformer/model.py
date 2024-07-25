import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math
from torch_scatter import scatter_mean


class Transformer_Finetuner(nn.Module):

    def __init__(self, config):
        super(Transformer_Finetuner, self).__init__()

        assert config['d_model'] % config['n_head'] == 0, "nheads must divide evenly into d_model"
        self.config = config
        self.d_model = config['d_model']

        # self.emb = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)

        self.pos_encoder = PositionalEncoding(
            d_model=config['d_model'],
            dropout=config['dropout'],
            vocab_size=config['vocab_size'],
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['n_head'],
            dim_feedforward=config['transformer_hidden_dim'],
            dropout=config['dropout'],
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_layers'],
        )

        self.mlp_input_dim = config['d_model']*2

        if self.config['add_att']:
            # self.cross_att_layer = CrossAttentionBlock(hidden_dim=config['d_model'], num_heads=config['att_nhead'], pooling=False)
            self.fusion_model = CrossAttention(config['d_model'], num_layers=config['num_layers'], num_heads=config['att_nhead'], batch_norm=False, activation="relu")
        
        self.predictor = MLP(num_layers=2, input_dim=self.mlp_input_dim, hidden_dim=config['mlp_hidden_dim'], output_dim=config['mlp_output_dim'])
        # self.predictor = ResNetClassifier(config['d_model']*2)


    def forward(self, x1, mask1, x2, mask2):

        h_total = []

        for x, mask in [(x1, mask1), (x2, mask2)]:
            mask = (1-mask).bool()
            x = x.transpose(0,1)

            # x = self.emb(x) * math.sqrt(self.d_model)
            x = x * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
            h = self.transformer_encoder(x, src_key_padding_mask=mask)
            # pool = h.mean(dim=0)
            if not self.config['add_att']:
                h = h[0,:,:]    # [512, 256]
            h_total.append(h)   
        
        if self.config['add_att']:
            # att11, att22, att12, att21, output_total = self.cross_att_layer(h_total[0].transpose(0,1), h_total[1].transpose(0,1), output_att=True)
            fusion_output = self.fusion_model(h_total[0], h_total[1], mask1, mask2)
            h_total = fusion_output
        else:
            h_total = torch.cat(h_total, dim=1) # [512, 512]
        # else:
        #     h_total = torch.cat(h_total, dim=-1)
            # h_total = h_total.transpose(0,1).transpose(1,2)
        
        out = self.predictor(h_total)

        return out
    

class Transformer_Pretrain(nn.Module):

    def __init__(self, config):
        super(Transformer_Pretrain, self).__init__()

        assert config['d_model'] % config['n_head'] == 0, "nheads must divide evenly into d_model"
        self.config = config
        self.d_model = config['d_model']
        self.random_init = config['random_init']
        
        # if self.random_init:
        #     self.emb = nn.Embedding(config['vocab_size']*9, config['d_model'], padding_idx=0)

        self.pos_encoder = PositionalEncoding(
            d_model=config['d_model'],
            dropout=config['dropout'],
            vocab_size=config['vocab_size'],
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['n_head'],
            dim_feedforward=config['transformer_hidden_dim'],
            dropout=config['dropout'],
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_layers'],
        )

        self.mlp_input_dim = config['d_model']
        
        self.predictor = MLP(num_layers=2, input_dim=self.mlp_input_dim, hidden_dim=config['mlp_hidden_dim'], output_dim=config['mlp_output_dim'])


    def forward(self, x, mask):

        mask = (1-mask).bool()
        x = x.transpose(0,1)

        if not self.random_init:
            x = x * math.sqrt(self.d_model)
        else:
            x = self.emb(x) * math.sqrt(self.d_model)
            
        x = self.pos_encoder(x)
        h = self.transformer_encoder(x, src_key_padding_mask=mask)
        h = h.mean(dim=0)
        # h = h[0,:,:]    # [512, 256]   
        
        out = self.predictor(h)

        return out



class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
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

        # self.pooling=pooling
        # if self.pooling:
        #     self.conv_layers = nn.ModuleList()
        #     for i in range(4):
        #         self.conv_layers.append(nn.Conv1d(in_channels=self.hidden_dim,out_channels=self.hidden_dim,
        #                                           kernel_size=2))

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

    def forward(self, input1, input2, mask1, mask2):

        # input1 = input1.transpose(0, 1)
        # input2 = input2.transpose(0, 1)
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

        return output1, output2
        
        # output = []
        # # [512, 201, 256]*2
        # for i, out in enumerate([self_out1, self_out2, cross_out1, cross_out2]):
        #     if self.pooling:
        #         # Conv1D
        #         output.append(self.conv_layers[i](out.transpose(1,2)).mean(dim=2).squeeze())
        #     else:
        #         output.append(out[:,0,:].squeeze())

        # output_total = torch.cat(output, dim=1) #[512, dim*4]

        # if not output_att:
        #     return output_total
        # else:
        #     att11 = nn.functional.softmax(logits11, dim=2)
        #     att22 = nn.functional.softmax(logits22, dim=2)
        #     att12 = nn.functional.softmax(logits12, dim=2)
        #     att21 = nn.functional.softmax(logits21, dim=2)
        #     return att11, att22, att12, att21, output_total
        

class CrossAttention(nn.Module):

    def __init__(self, hidden_dim=512, num_layers=1, num_heads=8, batch_norm=False, activation="relu"):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(CrossAttentionBlock(hidden_dim, num_heads))

        if batch_norm:
            self.protein_batch_norm_layers = nn.ModuleList()
            self.text_batch_norm_layers = nn.ModuleList()
            for _ in range(self.num_layers):
                self.protein_batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))
                self.text_batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def forward(self, protein_input, text_input, mask1, mask2, all_loss=None, metric=None):
        # Padding for protein inputs
        # protein_input, protein_mask = variadic_to_padded(protein_input, graph.num_residues, value=0)

        for i, layer in enumerate(self.layers):

            if i == 0:
                protein_input = protein_input.transpose(0, 1)
                text_input = text_input.transpose(0, 1)

            protein_input, text_input = layer(protein_input, text_input, mask1.bool(), mask2.bool())
            if self.batch_norm:
                protein_input = self.protein_batch_norm_layers[i](protein_input.transpose(1, 2)).transpose(1, 2)
                text_input = self.text_batch_norm_layers[i](text_input.transpose(1, 2)).transpose(1, 2)
            if self.activation:
                protein_input = self.activation(protein_input)
                text_input = self.activation(text_input)

        protein_output = scatter_mean(protein_input, mask1, dim=1)[:, 1:,].squeeze(1)
        text_output = scatter_mean(text_input, mask1, dim=1)[:, 1:,].squeeze(1)

        return torch.cat([protein_output, text_output], dim=-1)

        # return {
        #     "gene1": protein_output,
        #     "gene2": text_output
        # }



class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        
        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.linears.append(nn.Linear(input_dim, hidden_dim))
        for layer in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))

        for layer in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):

        h = x
        for i in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[i](self.linears[i](h)))
        return self.linears[-1](h)



class MLP_simple(nn.Module):

    def __init__(self, layer_size, output_dim, return_hidden=True):
        super(MLP_simple, self).__init__()

        self.return_hidden = return_hidden

        layers = []
        for i in range(len(layer_size)-1):
            layers.append(nn.Linear(layer_size[i], layer_size[i+1]))
            layers.append(nn.BatchNorm1d(layer_size[i+1]))
            layers.append(nn.ReLU())
        
        self.network = torch.nn.Sequential(*layers)
        self.lin = nn.Linear(layer_size[-1], output_dim)
        
    def forward(self, x):

        h = self.network(x)
        res = self.lin(torch.nn.functional.relu(h))
        if self.return_hidden:
            return h, res
        else:
            return res
        

