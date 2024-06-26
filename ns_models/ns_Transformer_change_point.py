import torch
import torch.nn as nn
from ns_layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from ns_layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
from ns_layers.LLSA import LLSA  # Import LLSA class
import sys


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class Model(nn.Module):
    """
    Non-stationary Transformer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

        # Initialize LLSA with CUSUM
        self.llsa = LLSA(num_features=configs.enc_in, eps=1e-5, affine=True, window_size=configs.seq_len,
                         hidden_size=10, change_point_threshold=0.1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_raw = x_enc.clone().detach()

        # Normalization using LLSA
        x_enc, mean_enc, std_enc = self.llsa(x_enc, mode='norm')
        assert not torch.isnan(x_enc).any(), f"encoder norm"
        # print(f"mean before cat: {mean_enc.shape}")
        mean_enc = torch.cat(mean_enc, dim=1)
        std_enc = torch.cat(std_enc, dim=1)
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        # print(x_enc)
        # Get statistics
        mean_enc_general = self.llsa.mean
        std_enc_general = self.llsa.stdev
        # print(f"mean: {mean_enc.shape}")
        # print(f"std: {std_enc.shape}")
        assert not torch.isnan(mean_enc).any(), f"mean nan"
        assert not torch.isnan(std_enc).any(), f"std nan"
        # print(mean_enc)
        # print(std_enc)
        tau = self.tau_learner(x_raw, std_enc_general).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc_general)  # B x S x E, B x 1 x E -> B x S
        assert not torch.isnan(tau).any(), f"tau nan"
        assert not torch.isnan(delta).any(), f"delta nan"

        # print(tau)
        # print(delta)
        # Model Inference
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)
        assert not torch.isnan(enc_out).any(), f"enc_out denorm"

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # print(f"delta: {delta.shape}")
        # print(f"tau: {tau.shape}")
        # De-normalization using LLSA
        assert not torch.isnan(dec_out).any(), f"before denorm"

        dec_out = dec_out * std_enc_general + mean_enc_general
        # print(dec_out)
        assert not torch.isnan(dec_out).any(), f"nan"

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]