import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from rouge import rouge_score
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv
from data_util import dataloader
from data_util.logging import logger
from fastNLP.core import seq_len_to_mask
from fastNLP.modules import LSTM, MLP

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


def is_infinite_dist(distss):
    is_nan_list = torch.isnan(distss)
    if (is_nan_list.max() == 1).item() == 1:
        return True
    return False


def init_lstm_wt(config, lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(config, linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


def init_wt_normal(config, wt):
    wt.data.normal_(std=config.trunc_norm_init_std)


def init_wt_unif(config, wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token], log_probs=self.log_probs + [log_prob], state=state, context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 heads,
                 num_hidden=256,
                 activation=F.elu,
                 feat_drop=0.1,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=True,
                 out_dim=None):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](g, h).flatten(1)
        # output layer mean of the attention head
        output = self.gat_layers[-1](g, h).mean(1)
        return output


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation=F.relu,
                 dropout=0.1,
                 out_dim=None):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h) + h  # residual
        return h


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(config, self.embedding.weight)
        self.join = nn.Linear(4 * config.hidden_dim, 2 * config.hidden_dim)
        init_linear_wt(config, self.join)
        self.lstm = LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.graph_feature_lstm = LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True,
                                       bidirectional=True)
        self.mlp = MLP(size_layer=[config.hidden_dim * 4, config.hidden_dim * 2, config.hidden_dim * 2, 1],
                       activation="tanh")

        self.criterion = nn.MSELoss(reduction="sum")

    def pooling(self, h_cnode_batch, encoder_outputs):
        config = self.config
        if config.pooling == "none":
            node_feature = h_cnode_batch
        elif config.pooling == "mean":
            node_feature = torch.mean(encoder_outputs, dim=1)
        elif config.pooling == "max":
            node_feature = torch.max(encoder_outputs, dim=1)[0]
        else:
            mean_feat = torch.mean(encoder_outputs, dim=1)
            max_feat = torch.max(encoder_outputs, dim=1)[0]
            feat = torch.cat([mean_feat, max_feat], 1)
            node_feature = self.join(feat)

        return node_feature.unsqueeze(1)

    # seq_lens should be in descending order
    def forward(self, enc_input, enc_lens, graph_enc_batch, nbr_inputs_len, nodes_num, graphs):
        config = self.config
        seq_lens = enc_lens
        # encode the input text
        embedded = self.embedding(enc_input)
        encoder_outputs, hidden = self.lstm(embedded, seq_lens)
        encoder_outputs = encoder_outputs.contiguous()  # B x t_k x 2*hidden_dim

        h_cnode_batch, _ = hidden
        h_cnode_batch = h_cnode_batch.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        # encode graph feature
        node_features = [self.pooling(h_cnode_batch, encoder_outputs)]
        neighbor_node_num = max(nodes_num - 1)

        for idx in range(neighbor_node_num):
            node_batch = graph_enc_batch[:, idx, :]
            len_batch = nbr_inputs_len[:, idx].clone()
            # there may be some error if  seq_len = 0 in this batch
            for i in range(len(len_batch)):
                len_batch[i] += (len_batch[i] == 0)
            node_embed = self.embedding(node_batch)

            if config.share_encoder:
                outputs, (h, c) = self.lstm(node_embed, len_batch)  # h =>2x batch x hidden_dim
            else:
                outputs, (h, c) = self.graph_feature_lstm(node_embed, len_batch)

            h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
            node_features.append(self.pooling(h_in, outputs))
        node_features = torch.cat(node_features, 1)  # batch x node_num x 2*hidden_dim
        # neighborhood  extraction
        if max(nodes_num) != 1 and config.sample_trick != "none":
            node_features_sample, nodes_num_sample, distribution, graphs_sample, idxs_sample = self.sample_graph(
                node_features,
                nodes_num,
                graphs)
        else:
            node_features_sample = node_features
            nodes_num_sample = nodes_num
            graphs_sample = graphs
            idxs_sample = None

        # delete padding nodes
        node_feature_res = []
        node_feature_idx = [0]
        for idx, node_feature in enumerate(node_features_sample):
            node_num = nodes_num_sample[idx]
            mask = torch.arange(node_num)
            node_feature_idx.append(node_feature_idx[-1] + len(mask))
            node_feature_res.append(torch.index_select(node_feature, 0, torch.tensor(mask, device=node_feature.device)))
        node_feature_res = torch.cat(node_feature_res, 0)
        assert len(node_feature_res) == sum(nodes_num).item()
        return encoder_outputs, hidden, node_feature_res, node_feature_idx, graphs_sample, idxs_sample

    def sample_graph(self, graph_feats, nodes_num, graphs):
        """
        neighborhood extraction module
        """
        source_node = graph_feats[:, 0, :]
        nbr_feats = graph_feats[:, 1:, :]
        graph_mask = seq_len_to_mask(nodes_num - 1, max_len=nbr_feats.shape[1])  # B x nbr_num

        batch_mask = (nodes_num != 1).float()[:, None, None].repeat(1, nbr_feats.shape[1],
                                                                      nbr_feats.shape[2])
        nbr_feats_clone = nbr_feats * batch_mask
        scores = self.salience_score(source_node, nbr_feats_clone)
        dist_ = F.softmax(scores, dim=1) * graph_mask.float()
        normalization_factor = dist_.sum(1, keepdim=True)
        normalization_factor += (normalization_factor == 0).float()

        dist = dist_ / normalization_factor  # B x node_num -1
        dist = dist.to(graph_feats.device)

        if max(nodes_num - 1) <= self.config.sample_num:
            b, n, h = nbr_feats.size()
            dist = dist.unsqueeze(-1).expand(b, n, h)
            nbr_feats = nbr_feats * dist
            return torch.cat([source_node.unsqueeze(1), nbr_feats], 1), nodes_num, dist, graphs, None
        else:
            graph_feats_sample, nodes_batch_sample, attn_dist_sample, graphs_sample, idxs_sample = self._sample_nbr(
                dist,
                nbr_feats,
                nodes_num,
                self.config.sample_num,
                graphs
            )
            b, n, h = graph_feats_sample.size()
            attn_dist_sample = attn_dist_sample.unsqueeze(-1).expand(b, n, h)
            graph_feats_sample = graph_feats_sample * attn_dist_sample

            return torch.cat([source_node.unsqueeze(1), graph_feats_sample],
                             1), nodes_batch_sample, attn_dist_sample, graphs_sample, idxs_sample


    def _sample_nbr(self, attn_dist, nbr_feats, nodes_num, sample_num, graphs):
        """
        sample sample_num nodes from  G_v depend on attn_dist
        """
        batch_num, max_nbr_num, hidden = nbr_feats.size()
        graph_feats_sample = torch.zeros(batch_num, sample_num, hidden).to(nbr_feats.device)
        attn_dist_sample = torch.zeros(batch_num, sample_num).to(nbr_feats.device)
        selected_idxs_sample = None
        graphs_sample = []
        for batch_idx in range(batch_num):
            distribution = attn_dist[batch_idx]
            graph_feat = nbr_feats[batch_idx]
            nbr_num = nodes_num[batch_idx] - 1
            if nbr_num > sample_num:
                _, topk_idxs = torch.topk(distribution, k=sample_num)

                if selected_idxs_sample is None:
                    selected_idxs_sample = topk_idxs

                selected_idxs = topk_idxs.detach().cpu().numpy()
                selected_idxs += 1
                selected_idxs = selected_idxs.tolist()
                selected_idxs.append(0)
                selected_idxs.sort()

                if self.training or batch_idx == 0:
                    g_full = graphs[batch_idx]
                    g = g_full.subgraph(selected_idxs)
                    g.detach_parent()

                    for node in g.nodes():
                        if node != 0 and not g.has_edge_between(0, node):
                            g.add_edge(0, node)
                            g.add_edge(node, 0)

                    graphs_sample.append(g)

                selected_idxs = torch.tensor(selected_idxs[1:], device=nbr_feats.device)
                selected_idxs -= 1
                graph_feats_sample[batch_idx] = torch.index_select(graph_feat, 0, selected_idxs)
                attn_dist_sample[batch_idx] = torch.index_select(distribution, 0, selected_idxs)
                if selected_idxs_sample is None:
                    selected_idxs_sample = selected_idxs
            else:
                graph_feats_sample[batch_idx] = graph_feat[:sample_num]
                attn_dist_sample[batch_idx] = distribution[:sample_num]
                graphs_sample.append(graphs[batch_idx])
            nodes_num[batch_idx] = min(sample_num + 1, nbr_num + 1)

        return graph_feats_sample, nodes_num, attn_dist_sample, graphs_sample, selected_idxs_sample

    def salience_score(self, hidden, graph_hiddens):
        hidden = hidden.unsqueeze(1).expand(hidden.shape[0], graph_hiddens.shape[1],
                                            hidden.shape[1]).contiguous()  # # b x t x h
        score = self.mlp(torch.cat([hidden, graph_hiddens], 2))
        return score.squeeze(2)  # [B*T]


class GNNEncoder(nn.Module):
    def __init__(self, config):
        super(GNNEncoder, self).__init__()
        heads = ([config.num_heads] * config.num_layers) + [config.num_out_heads]
        if config.GNN == "GAT":
            self.gnn = GAT(num_layers=config.num_layers, in_dim=config.hidden_dim * 2, heads=heads,
                           num_hidden=config.hidden_dim * 2, residual=config.residual)
        elif config.GNN == "GCN":
            self.gnn = GCN(in_feats=config.hidden_dim * 2, n_hidden=config.hidden_dim * 2,
                           n_layers=config.num_layers, dropout=config.gnn_drop)
        else:
            raise Exception("GNN not supported ")

    def forward(self, graphs, node_feats, node_idx, nodes_num_batch):
        # if graphs length = 1 there will be errors in dgl
        if len(graphs) == 1:
            graphs.append(dgl.DGLGraph())

        g = dgl.batch(graphs)
        if g.number_of_nodes() != len(node_feats):
            logger.error("error: number of nodes in dgl graph do not equal nodes in input graph !!!")
            logger.error(
                f"number of nodes this batch:{sum(nodes_num_batch).item()}, number of num in dgl graph {g.number_of_nodes()}")
            assert g.number_of_nodes() == len(node_feats)

        gnn_feat = self.gnn(g, node_feats)
        b = len(nodes_num_batch)
        n = max(nodes_num_batch)
        h = gnn_feat.shape[1]
        node_features = torch.zeros([b, n, h], device=gnn_feat.device)
        # 还原成 B x max_nodes_num x hidden
        for i in range(len(node_idx) - 1):
            curr_idx = node_idx[i]
            next_idx = node_idx[i + 1]
            mask = torch.arange(curr_idx, next_idx, device=gnn_feat.device)
            output_feat = torch.index_select(gnn_feat, 0, mask)
            if output_feat.shape[0] < n:
                pad_num = n - output_feat.shape[0]
                extra_zeros = torch.zeros(pad_num, h, device=gnn_feat.device)
                output_feat = torch.cat([output_feat, extra_zeros], 0)
            node_features[i] = output_feat

        return node_features


class ReduceState(nn.Module):
    def __init__(self, config):
        super(ReduceState, self).__init__()
        self.config = config
        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(config, self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(config, self.reduce_c)

    def forward(self, hidden):
        config = self.config
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))
        return hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)  # h, c dim = 1 x b x hidden_dim


class GraphAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GraphAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
        self.v = nn.Linear(self.hidden_size * 2, 1, bias=False)

    def forward(self, s_t, graph_feats, nodes_batch):
        graph_mask = seq_len_to_mask(nodes_batch)
        attn_energies = self.score(s_t, graph_feats)
        attn_dist_ = F.softmax(attn_energies, dim=1) * graph_mask.float()  # B  x T
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor
        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        graph_hidden = torch.bmm(attn_dist, graph_feats)  # Bx Tx H
        return graph_hidden.view(-1, self.hidden_size * 2), attn_dist.squeeze(1)

    def score(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1).expand(hidden.shape[0], encoder_outputs.shape[1], hidden.shape[1]).contiguous()
        # b x t x h
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = self.v(energy)
        return energy.squeeze(2)  # [B*T]


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.c_t_v_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage, c_t_v):
        config = self.config

        b, t_k, n = list(encoder_outputs.size())
        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        ctv_fea = self.c_t_v_proj(c_t_v)  # B x 2*hidden_dim
        ctv_fea_expanded = ctv_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        ctv_fea_expanded = ctv_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        # w_h *h + w_s *s_t + w_cv * c_t_v
        att_features = encoder_feature + dec_fea_expanded + ctv_fea_expanded  # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B x t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k
        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask.float()  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x 2*hidden_dim
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k
        if config.is_coverage:
            coverage = coverage.view(b, -1)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.attention_network = Attention(config)
        self.graph_atten_context = GraphAttention(config.hidden_dim)
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(config, self.embedding.weight)
        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True,
                            bidirectional=False)
        init_lstm_wt(config, self.lstm)
        self.p_gen_linear = nn.Linear(config.hidden_dim * 6 + config.emb_dim, 1)
        self.out1 = nn.Linear(config.hidden_dim * 5, config.emb_dim)  # p_vocab
        init_linear_wt(config, self.out1)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, gnn_feat, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step, nodes_batch):
        config = self.config
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

            c_t_v, graph_dist = self.graph_atten_context(s_t_hat, gnn_feat, nodes_batch)
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage, c_t_v)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)
        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

        c_t_v, graph_dist = self.graph_atten_context(s_t_hat, gnn_feat, nodes_batch)

        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage, c_t_v)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen_input = torch.cat((c_t, c_t_v, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        p_gen = self.p_gen_linear(p_gen_input)
        p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t, c_t_v), 1)  # B x hidden_dim * 5
        output = self.out1(output)  # B x hidden_dim
        output = torch.matmul(output, self.embedding.weight.transpose(0, 1))  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist
        if extra_zeros is not None:
            # B, total_vocab_size
            vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
        final_dist = vocab_dist_.scatter_add_(1, enc_batch_extend_vocab, attn_dist_)

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage, graph_dist


class CGSum(torch.nn.Module):
    def __init__(self, config, vocab=None):
        super(CGSum, self).__init__()
        self.use_cuda = config.use_gpu and torch.cuda.is_available()

        encoder = Encoder(config)
        decoder = Decoder(config)
        decoder.embedding.weight = encoder.embedding.weight

        reduce_state = ReduceState(config)
        self.config = config
        self.vocab = vocab

        self.mlp = MLP(size_layer=[config.hidden_dim * 2, config.hidden_dim * 2, config.hidden_dim * 2])
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

        self.encoder = encoder
        self.decoder = decoder
        self.gnnEncoder = GNNEncoder(config)
        self.reduce_state = reduce_state

    def get_input_from_batch(self, enc_input, enc_len, enc_input_extend_vocab, article_oovs, enc_len_mask=None):

        config = self.config
        device = enc_input.device
        batch_size = len(enc_len)
        extra_zeros = None

        if enc_len_mask is None:
            if self.training:
                enc_padding_mask = seq_len_to_mask(enc_len, config.max_enc_steps).float()
            else:
                enc_padding_mask = seq_len_to_mask(enc_len).float()
        else:
            enc_padding_mask = enc_len_mask.float()

        enc_batch_extend_vocab = enc_input_extend_vocab
        # max_art_oovs is the max length over all the article oov list in the batch
        max_art_oovs = 0
        for article_oov in article_oovs:
            if "N O N E" in article_oov:
                continue
            else:
                max_art_oovs = max(max_art_oovs, len(article_oov))
        if max_art_oovs > 0:
            extra_zeros = torch.zeros(batch_size, max_art_oovs).to(device)

        c_t_1 = torch.zeros(batch_size, 2 * config.hidden_dim).to(device)

        coverage = None

        if config.is_coverage:
            coverage = torch.zeros(enc_input.size()).to(enc_input.device)

        return enc_padding_mask, extra_zeros, enc_batch_extend_vocab, c_t_1, coverage

    def get_output_from_batch(self, dec_len):
        config = self.config
        dec_lens = dec_len
        max_dec_len = np.max(np.array(dec_lens.cpu()))
        dec_padding_mask = seq_len_to_mask(dec_lens, min(max_dec_len, config.max_dec_steps)).float().to(dec_len.device)
        return max_dec_len, dec_padding_mask

    def forward(self, enc_input, enc_len, enc_len_mask, graph, nbr_inputs, nbr_inputs_len, nodes_num, dec_input,
                dec_len, article_oovs, enc_input_extend_vocab):
        config = self.config
        graphs = graph
        enc_padding_mask, extra_zeros, enc_batch_extend_vocab, c_t_1, coverage = \
            self.get_input_from_batch(enc_input, enc_len, enc_input_extend_vocab, article_oovs, enc_len_mask)

        max_dec_len, dec_padding_mask = self.get_output_from_batch(dec_len)

        encoder_outputs, encoder_hidden, node_features, node_idx, graphs_sample, idxs_sample = self.encoder(
            enc_input,
            enc_len,
            nbr_inputs,
            nbr_inputs_len,
            nodes_num,
            graphs
        )

        neighbor_feat = self.gnnEncoder(graphs_sample, node_features, node_idx, nodes_num)
        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)
        (h_t, c_t) = self.reduce_state(encoder_hidden)
        s_t_1 = (h_t, c_t)
        list_final_dist = []
        list_coverage = []
        list_attn = []
        pred = None

        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_input[:, di]  # Teacher forcing
            # init state: s_t_1 from encoder, c_t_1  with 0
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage, graph_dist = self.decoder(y_t_1, s_t_1,
                                                                                                 encoder_outputs,
                                                                                                 encoder_feature,
                                                                                                 neighbor_feat,
                                                                                                 enc_padding_mask,
                                                                                                 c_t_1,
                                                                                                 extra_zeros,
                                                                                                 enc_batch_extend_vocab,
                                                                                                 coverage, di,
                                                                                                 nodes_num)

            _, max_position = torch.max(final_dist, 1)
            max_position = max_position.unsqueeze(1)
            if pred is None:
                pred = max_position
            else:
                pred = torch.cat((pred, max_position), 1)

            list_final_dist.append(final_dist)
            list_attn.append(attn_dist)
            if config.is_coverage:
                list_coverage.append(coverage)
                coverage = next_coverage
        return {"pred": pred, "list_final_dist": list_final_dist, "list_coverage": list_coverage,
                "list_attn": list_attn, "max_dec_len": max_dec_len, "dec_padding_mask": dec_padding_mask,
                "dec_lens_var": dec_len}

    def unpadding(self, enc_len, enc_input, enc_input_extend_vocab):
        return enc_input[:enc_len], enc_input_extend_vocab[:enc_len]

    def decode(self, enc_len, enc_input, nbr_inputs, nbr_inputs_len, graph_single, node_num, article_oovs,
               enc_input_extend_vocab):
        """
        inference procedure including beam search and rouge credit
        """
        config = self.config
        enc_input, enc_input_extend_vocab = self.unpadding(enc_len, enc_input, enc_input_extend_vocab)
        enc_input = enc_input.unsqueeze(0).expand(config.beam_size, list(enc_input.size())[0]).contiguous()
        enc_len = enc_len.unsqueeze(0).expand(config.beam_size).contiguous()
        enc_input_extend_vocab = enc_input_extend_vocab.unsqueeze(0).expand(config.beam_size,
                                                                            list(enc_input_extend_vocab.size())[
                                                                                0]).contiguous()
        nbr_inputs_len = nbr_inputs_len.unsqueeze(0).repeat(config.beam_size, 1)
        nbr_inputs = nbr_inputs.unsqueeze(0).repeat(config.beam_size, 1, 1)
        node_num = node_num.unsqueeze(0).repeat(config.beam_size)
        graphs = [graph_single]
        enc_padding_mask, extra_zeros, enc_batch_extend_vocab, c_t_0, coverage_t_0 = \
            self.get_input_from_batch(enc_input, enc_len, enc_input_extend_vocab, [article_oovs])
        encoder_outputs, encoder_hidden, node_features, node_idx, graphs_sample, idxs_sample = self.encoder(
            enc_input,
            enc_len,
            nbr_inputs,
            nbr_inputs_len,
            node_num,
            graphs
        )
        feature_idx = int(len(node_features) / config.beam_size)
        neighbor_feat = self.gnnEncoder(graphs_sample, node_features[0:feature_idx], [0, feature_idx], [node_num[0]])
        neighbor_feat = neighbor_feat.repeat(config.beam_size, 1, 1)

        if node_num[0] > 1:
            nbrs = nbr_inputs[0]
            if idxs_sample is not None:
                nbrs = torch.index_select(nbrs, 0, idxs_sample)
        else:
            nbrs = None

        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        (h_t, c_t) = self.reduce_state(encoder_hidden)
        s_t_0 = (h_t, c_t)
        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.to_index(dataloader.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None)) for _ in range(config.beam_size)]
        results = []
        steps = 0
        rouge_every_n = 5
        stop_idx = self.vocab.to_index(dataloader.STOP_DECODING)
        eos_idx = self.vocab.to_index(".")
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < len(self.vocab) else self.vocab.to_index(dataloader.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = torch.LongTensor(latest_tokens).to(dec_h.device)

            all_state_h = []
            all_state_c = []
            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t, graph_dist = self.decoder(y_t_1, s_t_1,
                                                                                          encoder_outputs,
                                                                                          encoder_feature,
                                                                                          neighbor_feat,
                                                                                          enc_padding_mask,
                                                                                          c_t_1,
                                                                                          extra_zeros,
                                                                                          enc_batch_extend_vocab,
                                                                                          coverage_t_1,
                                                                                          steps,
                                                                                          node_num)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)
            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()
            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            attn_dists = graph_dist[0][1:]
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    tokens = new_beam.tokens
                    cw = config.constant_weight
                    # rouge credit
                    if config.constant_weight != 0 and nbrs is not None and (steps + 1) % rouge_every_n == 0:
                        max_idx = torch.argmax(attn_dists)
                        tmp = nbrs[max_idx].cpu().numpy().tolist()
                        reference = list(map(str, tmp))[:(steps + 1)]
                        prediction = list(map(str, tokens[1:]))
                        reference_text = " ".join(reference)
                        reference = reference_text.split(f" {eos_idx} ")
                        prediction_text = " ".join(prediction)
                        prediction = prediction_text.split(f" {eos_idx} ")
                        score = rouge_score.rouge_n(prediction, reference, 1)["f"]
                        if config.f_t == "exp":
                            rd_score = score if steps < config.l_s else np.exp(1 - config.l_s / steps) * score
                        else:
                            rd_score = score

                        new_beam.log_probs[-1] += cw * steps * rd_score

                    # trigram blocking
                    if len(tokens) > 3 and self.config.trigram_blocking:
                        trigrams = [(tokens[i - 1], tokens[i], tokens[i + 1]) for i in range(1, len(tokens) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            new_beam.log_probs[-1] = -10e20
                        if tokens[-1] == tokens[-2]:
                            new_beam.log_probs[-1] -= 10e5
                    all_beams.append(new_beam)

            beams = []
            for h in CGSum.sort_beams(all_beams):
                if h.latest_token == stop_idx:
                    # only save paths more than min decoding steps
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)
        output_ids = [int(t) for t in beams_sorted[0].tokens[1:]]

        return output_ids

    def predict(self, enc_input, enc_len, nbr_inputs, nbr_inputs_len, graph, nodes_num, article_oovs,
                enc_input_extend_vocab):
        """
        tester will call this method instead of forward
        """
        output_ids = []
        batch_size, seq_len = list(enc_input.size())
        # decode the target summary with batch_size = 1
        for _num in range(batch_size):
            graph_i = graph[_num]
            nbr_inputs_len_i = nbr_inputs_len[_num]
            nbr_inputs_i = nbr_inputs[_num]
            nodes_num_i = nodes_num[_num]
            assert nodes_num_i == graph_i.number_of_nodes()
            enc_len_i = enc_len[_num]
            enc_input_i = enc_input[_num]
            article_oovs_i = article_oovs[_num]
            enc_input_extend_vocab_i = enc_input_extend_vocab[_num]
            pred = self.decode(enc_len_i, enc_input_i, nbr_inputs_i, nbr_inputs_len_i, graph_i,
                               nodes_num_i,
                               article_oovs_i,
                               enc_input_extend_vocab_i)
            output_ids.append(pred)
        return {"prediction": output_ids}

    @staticmethod
    def sort_beams(beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

