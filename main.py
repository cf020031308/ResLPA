import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.nn


if len(sys.argv) > 3:
    g_method, g_data, g_622 = sys.argv[1:]
    g_622 = (g_622.lower() not in '0 f false')
else:
    g_method = 'reslpa'
    g_data = 'cora'
    g_622 = True

gpu = lambda x: x
if (
        torch.cuda.is_available()
        # Diable GPU when using dgl due to cuda version
        and g_method not in ('sage', 'gat', 'gat-large')
        # Coauthor-Physics is too large
        # for AdaLPA, GCN-LPA, ResLPA and Fast ResLPA
        and not (
            g_data -= 'coauthor-phy'
            and g_method in ('adalpa', 'gcnlpa', 'reslpa', 'fastreslpa')
        )
):
    dev = torch.device('cuda:3')
    gpu = lambda x: x.to(dev)


def optimize(params):
    print('params:', sum(p.numel() for p in params))
    return optim.Adam(params, lr=0.01)


def speye(n):
    return torch.sparse_coo_tensor(
        torch.arange(n).view(1, -1).repeat(2, 1), [1] * n)


def spnorm(A, eps=1e-5):
    D = (torch.sparse.sum(A, dim=1).to_dense() + eps) ** -0.5
    indices = A._indices()
    return torch.sparse_coo_tensor(indices, D[indices[0]] * D[indices[1]])


def count_subgraphs(adj):
    n_nodes = adj.shape[0]
    A = adj + gpu(torch.eye(n_nodes))
    mask = gpu(torch.zeros(n_nodes, dtype=bool))
    next_mask = mask.clone()
    count = 0
    while mask.sum() < n_nodes:
        next_mask[torch.arange(n_nodes)[~mask][0]] = True
        count += 1
        while mask.sum() < next_mask.sum():
            mask = next_mask
            next_mask = A[:, mask].sum(dim=1) > 0
    return count


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(self.__class__, self).__init__()
        self.conv1 = dgl.nn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dgl.nn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.leaky_relu(h)
        h = self.conv2(graph, h)
        return h


class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, A=None):
        super(self.__class__, self).__init__()
        self.conv1 = gpu(nn.Linear(in_feats, hid_feats))
        self.conv2 = gpu(nn.Linear(hid_feats, out_feats))
        self.A = A

    def forward(self, feats):
        h = self.conv1(self.A @ feats)
        h = F.leaky_relu(h)
        h = self.conv2(self.A @ h)
        return h


class AdaLPA(nn.Module):
    def __init__(self, src, dst, n_nodes, n_labels, props=5):
        super(self.__class__, self).__init__()
        self.adj_i = gpu(torch.cat((src.view(1, -1), dst.view(1, -1)), dim=0))
        self.adj_v = nn.Parameter(gpu(torch.ones(src.shape[0])))
        self.props = props
        self.shape = (n_nodes, n_labels)

    def forward(self, mask, labels):
        self.A = torch.sparse.softmax(
            gpu(torch.sparse_coo_tensor(self.adj_i, self.adj_v)),
            dim=1)
        A = self.A.to_dense()
        Y = gpu(torch.zeros(self.shape))
        for _ in range(self.props):
            Y[mask] = labels
            Y = A @ Y
        return Y


class GAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, head=2):
        super(self.__class__, self).__init__()
        self.conv1 = dgl.nn.GATConv(
            in_feats=in_feats, out_feats=hid_feats, num_heads=head)
        self.conv2 = dgl.nn.GATConv(
            in_feats=hid_feats * head, out_feats=out_feats, num_heads=1)

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.leaky_relu(h)
        h = h.flatten(1)
        h = self.conv2(graph, h)
        return h.squeeze()


class MLP(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(self.__class__, self).__init__()
        self.pred = gpu(nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, out_feats),
        ))

    def forward(self, x):
        return self.pred(x)


class Res(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(self.__class__, self).__init__()
        self.dec = gpu(nn.Linear(in_feats, hid_feats))
        self.pred = gpu(nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(2 * hid_feats, out_feats),
            nn.Tanh(),
        ))
        self.w = nn.Parameter(gpu(torch.ones(1, out_feats)))
        self.b = nn.Parameter(gpu(torch.zeros(1, out_feats)))

    def forward(self, x, y):
        h = torch.cat((x, y), dim=-1)
        return self.pred(h)
        return self.w * self.pred(h) + self.b


graph = (
    dgl.data.CoraGraphDataset()[0] if g_data == 'cora'
    else dgl.data.CiteseerGraphDataset()[0] if g_data == 'citeseer'
    else dgl.data.PubmedGraphDataset()[0] if g_data == 'pubmed'
    else dgl.data.CoauthorCSDataset()[0] if g_data == 'coauthor-cs'
    else dgl.data.CoauthorPhysicsDataset()[0] if g_data == 'coauthor-phy'
    else None
)
# print(count_subgraphs(graph.adj().to_dense()))
# exit()
if g_data.startswith('coauthor'):
    # Fix bug of https://github.com/dmlc/dgl/issues/2553 @20210121
    src, _ = graph.edges()
    e = src.shape[0] // 2
    g = dgl.graph((src[:e], src[e:]))
    g.ndata.update(graph.ndata)
    graph = g
n_nodes = graph.num_nodes()
node_features = gpu(graph.ndata['feat'])
node_labels = graph.ndata['label']
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)
src, dst = graph.edges()
dsrc = torch.cat((src, torch.arange(n_nodes)))
ddst = torch.cat((dst, torch.arange(n_nodes)))


class Stat(object):
    def __init__(self, name='', statfrom=0):
        self.name = name
        self.accs = []
        self.best_accs = []
        self.times = []
        self.statfrom = statfrom

    def __call__(self, logits, startfrom=0):
        self.accs.append([
            ((logits[mask].cpu().max(dim=1).indices == node_labels[mask]).sum()
             / mask.sum().float()).item()
            for mask in (valid_mask, test_mask)
        ])

    def start_run(self):
        self.tick = time.time()

    def end_run(self):
        self.times.append(time.time() - self.tick)
        if len(self.accs) < self.statfrom:
            print('Not enough scores.')
            return
        self.accs = torch.tensor(self.accs[self.statfrom:])
        self.best_accs.append(
            self.accs[self.accs.max(dim=0).indices[0], 1])
        self.accs = []
        print('best:', self.best_accs[-1].item())

    def end_all(self):
        acc = 100 * torch.tensor(self.best_accs)
        tm = torch.tensor(self.times)
        print(self.name)
        print('time:%.3f±%.3f' % (tm.mean().item(), tm.std().item()))
        print('acc:%.2f±%.2f' % (acc.mean().item(), acc.std().item()))


evaluate = Stat(
    name='data: %s, method: %s, 622: %s' % (g_data, g_method, g_622))
for run in range(10):
    torch.manual_seed(run)
    if g_622:
        train_mask = torch.zeros(n_nodes, dtype=bool)
        valid_mask = torch.zeros(n_nodes, dtype=bool)
        test_mask = torch.zeros(n_nodes, dtype=bool)
        idx = torch.randperm(n_nodes)
        val_num = int(0.2 * n_nodes)
        test_num = int(0.2 * n_nodes)
        train_mask[idx[val_num + test_num:]] = True
        valid_mask[idx[:val_num]] = True
        test_mask[idx[val_num:val_num + test_num]] = True
    else:
        train_mask = graph.ndata['train_mask']
        valid_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
    train_labels = gpu(node_labels[train_mask])
    evaluate.start_run()
    if g_method in ('sage', 'gat', 'gat-large'):
        evaluate.statfrom = 5
        gnn = gpu(
            SAGE(in_feats=n_features,
                 hid_feats=32,
                 out_feats=n_labels) if g_method == 'sage'
            else GAT(in_feats=n_features,
                     hid_feats=32,
                     head=2,
                     out_feats=n_labels) if g_method == 'gat'
            else GAT(in_feats=n_features,
                     hid_feats=64,
                     head=4,
                     out_feats=n_labels)
        )
        opt = optimize([*gnn.parameters()])
        for _ in range(200):
            opt.zero_grad()
            Y = gnn(graph, node_features)
            loss = F.cross_entropy(Y[train_mask], train_labels)
            loss.backward()
            opt.step()
            evaluate(Y)
    elif g_method in ('mlp', 'gcn'):
        evaluate.statfrom = 5
        if g_method == 'mlp':
            model = MLP(n_features, 64, n_labels)
        else:
            A = spnorm(gpu(graph.adj() + speye(n_nodes)), eps=0)
            model = GCN(n_features, 64, n_labels, A)
        opt = optimize([*model.parameters()])
        for _ in range(200):
            opt.zero_grad()
            Y = model(node_features)
            loss = F.cross_entropy(Y[train_mask], train_labels)
            loss.backward()
            opt.step()
            evaluate(Y)
    else:
        n_train = int(train_mask.sum())
        train_probs = gpu(torch.zeros(n_train, n_labels))
        train_probs[torch.arange(n_train), train_labels] = 1
        if g_method == 'cs':
            mlp = MLP(in_feats=n_features, hid_feats=64, out_feats=n_labels)
            opt = optimize([*mlp.parameters()])
            best_acc = 0
            for _ in range(200):
                opt.zero_grad()
                probs = torch.softmax(mlp(node_features), dim=-1)
                loss = F.cross_entropy(probs[train_mask], train_labels)
                loss.backward()
                opt.step()
                _, indices = torch.max(probs[valid_mask].cpu(), dim=1)
                acc = (
                    (indices == node_labels[valid_mask]).sum()
                    / valid_mask.sum().float()
                ).item()
                if acc > best_acc:
                    best_acc = acc
                    Y = probs
            dy = train_probs - Y[train_mask]
            dY = gpu(torch.zeros(n_nodes, n_labels))
            alpha = 0.5
            A = spnorm(gpu(graph.adj()))
            for _ in range(50):
                dY[train_mask] = dy
                dY = (1 - alpha) * dY + alpha * (A @ dY)
            Y += dY
            Y[train_mask] = train_probs
            alpha = 0.1
            for _ in range(50):
                Y = (1 - alpha) * Y + alpha * (A @ Y)
            evaluate(Y)
        elif g_method == 'adalpa':
            evaluate.statfrom = 5
            lpa = AdaLPA(dsrc, ddst, n_nodes, n_labels, props=5)
            opt = optimize([*lpa.parameters()])
            for _ in range(200):
                opt.zero_grad()
                Y = lpa(train_mask, train_probs)
                loss = F.cross_entropy(Y[train_mask], train_labels)
                loss.backward()
                opt.step()
                evaluate(Y)
        elif g_method == 'gcnlpa':
            evaluate.statfrom = 5
            gcn = GCN(n_features, 64, n_labels)
            lpa = AdaLPA(dsrc, ddst, n_nodes, n_labels, props=5)
            opt = optimize([*gcn.parameters(), *lpa.parameters()])
            for _ in range(200):
                opt.zero_grad()
                _Y = lpa(train_mask, train_probs)
                gcn.A = lpa.A.detach()
                Y = gcn(node_features)
                loss = (
                    F.cross_entropy(Y[train_mask], train_labels)
                    + 1e-4 * F.cross_entropy(_Y[train_mask], train_labels))
                loss.backward()
                opt.step()
                evaluate(Y)
        elif g_method == 'fastreslpa':
            evaluate.statfrom = 5
            f = Res(n_features, 64, n_labels)
            opt = optimize([*f.parameters()])
            alpha = 0.9
            A = gpu(graph.adj())
            D = (torch.sparse.sum(A, dim=1) ** -1).to_dense().unsqueeze(-1)
            Y = gpu(torch.zeros(n_nodes, n_labels))
            for _ in range(50):
                opt.zero_grad()
                feats = f.dec(node_features)
                res = f(feats[src], feats[dst])
                Rs = torch.cat([
                    torch.sparse.sum(
                        torch.sparse_coo_tensor(A._indices(), res[:, c]),
                        dim=-1
                    ).to_dense().unsqueeze(-1)
                    for c in range(n_labels)
                ], dim=1)
                Y[train_mask] = train_probs
                Y = Y / (Y.norm(dim=1, keepdim=True) + 1e-5)
                Z = (1 - alpha) * Y + alpha * (A @ Y + Rs) * D
                loss = F.cross_entropy(Z[train_mask], train_labels)
                loss.backward()
                opt.step()
                Y = Z.detach()
                evaluate(Y)
        elif g_method == 'reslpa':
            evaluate.statfrom = 5
            f = Res(n_features, 64, n_labels)
            opt = optimize([*f.parameters()])
            alpha = 0.9
            A = gpu(graph.adj())
            D = (torch.sparse.sum(A, dim=1) ** -1).to_dense().unsqueeze(-1)
            for _ in range(200):
                opt.zero_grad()
                feats = f.dec(node_features)
                res = f(feats[src], feats[dst])
                Rs = torch.cat([
                    torch.sparse.sum(
                        torch.sparse_coo_tensor(A._indices(), res[:, c]),
                        dim=-1
                    ).to_dense().unsqueeze(-1)
                    for c in range(n_labels)
                ], dim=1)
                Y = gpu(torch.zeros(n_nodes, n_labels))
                for _ in range(5):
                    Y[train_mask] = train_probs
                    Y = Y / (Y.norm(dim=1, keepdim=True) + 1e-5)
                    Y = (1 - alpha) * Y + alpha * (A @ Y + Rs) * D
                Y = Y / (Y.norm(dim=1, keepdim=True) + 1e-5)
                Y = (1 - alpha) * Y + alpha * (A @ Y + Rs) * D
                loss = F.cross_entropy(Y[train_mask], train_labels)
                loss.backward()
                opt.step()
                for _ in range(20):
                    Y[train_mask] = train_probs
                    Y = Y / (Y.norm(dim=1, keepdim=True) + 1e-5)
                    Y = (1 - alpha) * Y + alpha * (A @ Y + Rs) * D
                evaluate(Y)
        else:
            A = torch.sparse.softmax(gpu(graph.adj()), dim=1)
            alpha = 0.4
            Y = gpu(torch.zeros(n_nodes, n_labels))
            for _ in range(50):
                Y[train_mask] = train_probs
                Y = (1 - alpha) * Y + alpha * (A @ Y)
                evaluate(Y)
    evaluate.end_run()
evaluate.end_all()
