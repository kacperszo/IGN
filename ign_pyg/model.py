"""IGN on plain PyTorch, parameter-compatible with the published DGL checkpoints.

Why this exists: IGN's only working configuration runs on torch 1.3.1 with dgl 0.4.3. Every
attempt to lift it has failed — the `dgl 2.1.0` "modern" tier returns R=-0.155 against the
reference tier's 0.802, and torch 2.4 with dgl 2.4.0+cu124, the only CUDA pairing data.dgl.ai
actually serves, is no better. DGL has had no commits in a year, publishes its CUDA wheels
under torch versions they cannot run on, and pip silently substitutes the CPU build when its
index is unreachable. Porting off DGL removes the whole class of problem.

Module and parameter names are kept identical to `scripts/model_v2.py`, so the published
checkpoints load without a key mapping — and so a divergence can only be arithmetic.

The graph is carried the way PyG carries one — an `edge_index` pair plus feature tensors —
but nothing here imports PyG or `torch_scatter`. The two operations it needs are
written against torch itself, so it runs on any modern torch with no extension wheel to
match against it. That is the entire point: DGL's packaging is what made this port necessary,
and reproducing the dependency in a different colour would carry the problem across.

The DGL operations this replaces, and what they become:

    edge_softmax(g, logits)              softmax(logits, dst, num_nodes=N)
    update_all(copy_e, sum)              scatter_add(e, dst)
    update_all(u_mul_e('hv','a'), sum)   scatter_add(hv[src] * a, dst)
    apply_edges(cat[src, dst])           direct indexing on edge_index
    u_add_v('h', 'h', 'm')               h[src] + h[dst]
    dgl.sum_edges(g, 'e', 'w')           scatter_add(e * w, edge_batch)

DGL aggregates messages at the destination and PyG's convention is edge_index[0] = source,
edge_index[1] = destination, so `dst` below is always `edge_index[1]`. Getting that backwards
produces a model that trains and predicts and is quietly wrong, which is the failure this port
exists to avoid repeating.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: int | None = None) -> torch.Tensor:
    """Sum `src` rows into `dim_size` buckets given by `index` — DGL's `sum` aggregation.

    Written against plain torch rather than imported from `torch_scatter`. That package is a
    compiled extension pinned to an exact torch build, which is the same packaging trap this
    port exists to escape; `index_add_` is in torch itself and needs no wheel index.
    """
    size = list(src.shape)
    size[dim] = int(dim_size) if dim_size is not None else int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.index_add_(dim, index, src)


def softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax over the edges sharing a destination — DGL's `edge_softmax`.

    The per-destination maximum is subtracted first. That is not cosmetic: attention logits
    here come out of a LeakyReLU with no bound, and exponentiating them directly overflows to
    inf and then to nan, which would show up as a model that trains for a while and then
    stops.
    """
    src = src.float()
    peak = src.new_full((num_nodes,) + src.shape[1:], float("-inf"))
    # scatter_reduce_ rather than index_reduce_: the two agree exactly here, and only the
    # latter prints a beta warning on every fresh process.
    peak = peak.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src), src,
                                reduce="amax", include_self=True)
    out = (src - peak[index]).exp()
    denom = scatter_add(out, index, dim=0, dim_size=num_nodes)
    return out / (denom[index] + 1e-16)


class FC(nn.Module):
    """The affinity head: a stack of linear layers over the pooled edge representation."""

    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super().__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        # Two separate `if`s, not `if/elif` — copied from the original and kept deliberately.
        # For j == 0 both run, so the first pass appends its own block and then falls into
        # the `else` and appends a second one; n_FC_layer=2 therefore builds three Linear
        # layers, not two. Almost certainly a slip by the authors, but it is the shape the
        # published weights were trained into, so tidying it here would load a checkpoint
        # into a different network. Writing `elif` did exactly that, and the checkpoint
        # refused to load — which is the only reason it was caught.
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h


class AttentiveGRU1(nn.Module):
    """Edge features into node features, weighted by attention, mixed with a GRU."""

    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super().__init__()
        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size),
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, edge_index, num_nodes, edge_logits, edge_feats, node_feats):
        dst = edge_index[1]
        e = softmax(edge_logits, dst, num_nodes=num_nodes) * self.edge_transform(edge_feats)
        context = F.elu(scatter_add(e, dst, dim=0, dim_size=num_nodes))
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    """Neighbour node features into node features, weighted by attention, mixed with a GRU."""

    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super().__init__()
        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size),
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, edge_index, num_nodes, edge_logits, node_feats):
        src, dst = edge_index[0], edge_index[1]
        a = softmax(edge_logits, dst, num_nodes=num_nodes)
        hv = self.project_node(node_feats)
        context = F.elu(scatter_add(hv[src] * a, dst, dim=0, dim_size=num_nodes))
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    """The first AttentiveFP layer, the only one that reads edge features."""

    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super().__init__()
        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU(),
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU(),
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU(),
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def forward(self, edge_index, num_nodes, node_feats, edge_feats):
        src, dst = edge_index[0], edge_index[1]
        hv_new = self.project_node(node_feats)
        # apply_edges1: source node features concatenated with the edge's own features
        he1 = self.project_edge1(torch.cat([node_feats[src], edge_feats], dim=1))
        # apply_edges2: destination's *projected* features concatenated with the above
        logits = self.project_edge2(torch.cat([hv_new[dst], he1], dim=1))
        return self.attentive_gru(edge_index, num_nodes, logits, he1, hv_new)


class GNNLayer(nn.Module):
    """Subsequent AttentiveFP layers, which see only node features."""

    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super().__init__()
        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU(),
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def forward(self, edge_index, num_nodes, node_feats):
        src, dst = edge_index[0], edge_index[1]
        # note the order: destination first, then source — matching apply_edges upstream
        logits = self.project_edge(torch.cat([node_feats[dst], node_feats[src]], dim=1))
        return self.bn_layer(self.attentive_gru(edge_index, num_nodes, logits, node_feats))


class ModifiedAttentiveFPGNNV2(nn.Module):
    """AttentiveFP over the covalent graph, summing the output of every layer."""

    def __init__(self, node_feat_size, edge_feat_size, num_layers=2,
                 graph_feat_size=200, dropout=0.):
        super().__init__()
        self.init_context = GetContext(node_feat_size, edge_feat_size,
                                       graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList(
            GNNLayer(graph_feat_size, graph_feat_size, dropout)
            for _ in range(num_layers - 1)
        )

    def forward(self, edge_index, num_nodes, node_feats, edge_feats):
        node_feats = self.init_context(edge_index, num_nodes, node_feats, edge_feats)
        # the running sum is the "V2" modification: the authors accumulate every layer's
        # output rather than returning only the last
        sum_node_feats = node_feats
        for gnn in self.gnn_layers:
            node_feats = gnn(edge_index, num_nodes, node_feats)
            sum_node_feats = sum_node_feats + node_feats
        return sum_node_feats


class ModifiedAttentiveFPPredictorV2(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers=2,
                 graph_feat_size=200, dropout=0.):
        super().__init__()
        self.gnn = ModifiedAttentiveFPGNNV2(node_feat_size, edge_feat_size,
                                            num_layers, graph_feat_size, dropout)
        self.predict = nn.Sequential(nn.Dropout(dropout), nn.Linear(graph_feat_size, 1))

    def forward(self, edge_index, num_nodes, node_feats, edge_feats):
        return self.gnn(edge_index, num_nodes, node_feats, edge_feats)


class DTIConvGraph3(nn.Module):
    """The interaction graph: edge state from its endpoints' node features plus itself."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mpl = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim), nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim), nn.LeakyReLU(),
        )

    def forward(self, edge_index, atom_feats, bond_feats):
        src, dst = edge_index[0], edge_index[1]
        m = atom_feats[src] + atom_feats[dst]          # dgl.function.u_add_v
        return self.mpl(torch.cat([bond_feats, m], dim=1))


class DTIConvGraph3Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.grah_conv = DTIConvGraph3(in_dim, out_dim)   # spelling kept: checkpoint key
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, edge_index, atom_feats, bond_feats):
        new_feats = self.grah_conv(edge_index, atom_feats, bond_feats)
        return self.bn_layer(self.dropout(new_feats))


class EdgeWeightAndSum_V2(nn.Module):
    """Weighted sum over an interaction graph's edges, one vector per complex."""

    def __init__(self, in_feats):
        super().__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(nn.Linear(in_feats, 1), nn.Sigmoid())

    def forward(self, edge_feats, edge_batch, num_graphs):
        weights = self.atom_weighting(edge_feats)
        h_g_sum = scatter_add(edge_feats * weights, edge_batch, dim=0, dim_size=num_graphs)
        return h_g_sum, weights


class IGN(nn.Module):
    """Covalent AttentiveFP, then an interaction graph, then a weighted edge readout.

    `encode` stops at the pooled representation, which is what the benchmark freezes; `FC` is
    the affinity head. That boundary is the authors' own — the readout produces a genuine
    complex-level vector, unlike the models here whose head runs per edge.
    """

    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size,
                 outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super().__init__()
        self.cov_graph = ModifiedAttentiveFPPredictorV2(
            node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size + 1, outdim_g3, dropout)
        self.FC = FC(outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks)
        self.readout = EdgeWeightAndSum_V2(outdim_g3)

    def encode(self, cov_edge_index, num_nodes, atom_feats, bond_feats,
               inter_edge_index, inter_edge_feats, edge_batch, num_graphs):
        atom_feats = self.cov_graph(cov_edge_index, num_nodes, atom_feats, bond_feats)
        bond_feats3 = self.noncov_graph(inter_edge_index, atom_feats, inter_edge_feats)
        h_g_sum, weights = self.readout(bond_feats3, edge_batch, num_graphs)
        return h_g_sum, weights

    def forward(self, cov_edge_index, num_nodes, atom_feats, bond_feats,
                inter_edge_index, inter_edge_feats, edge_batch, num_graphs):
        h_g_sum, weights = self.encode(cov_edge_index, num_nodes, atom_feats, bond_feats,
                                       inter_edge_index, inter_edge_feats,
                                       edge_batch, num_graphs)
        return self.FC(h_g_sum), weights
