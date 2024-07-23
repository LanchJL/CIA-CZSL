import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MLP
import numpy as np
from .embeddings import _Py, _Vs_Vo, _Ps_Po



class CZSL(nn.Module):
    def __init__(self, dset, args):
        super(CZSL, self).__init__()
        self.args = args
        self.dset = dset

        self.device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(self.device)
            objs = torch.LongTensor(objs).to(self.device)
            pairs = torch.LongTensor(pairs).to(self.device)
            return attrs, objs, pairs
        def remove_from_B(A, B):
            A_set = set(A.tolist())
            B_set = set(B.tolist())
            new_B_set = B_set - A_set
            new_B = torch.tensor(list(new_B_set), dtype=torch.long)
            return new_B
        def union_of_AB(A, B):
            combined = torch.cat([A, B], dim=0)
            union = torch.unique(combined)
            return union
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(self.device), \
                                          torch.arange(len(self.dset.objs)).long().to(self.device)
        self.scale = args.tem
        self.train_forward = self.train_forward_closed
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs
        try:
            self.args.fc_emb = self.args.fc_emb.split(',')
        except:
            self.args.fc_emb = [self.args.fc_emb]
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(self.device), \
                                          torch.arange(len(self.dset.objs)).long().to(self.device)
        self.pairs = dset.pairs

        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)
        if args.nlayers:
            self.image_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=args.relu, num_layers=args.nlayers,
                                      dropout=self.args.dropout,
                                      norm=self.args.norm, layers=layers)
        # Fixed
        self.composition = args.composition
        if self.args.train_only:
            train_idx = []
            test_idx = []
            val_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current])
            self.train_idx = torch.LongTensor(train_idx).to(self.device)

            for current in dset.val_pairs:
                val_idx.append(dset.all_pair2idx[current])
            self.val_idx = torch.LongTensor(val_idx).to(self.device)
            self.val_idx = remove_from_B(self.train_idx,self.val_idx)

            for current in dset.test_pairs:
                test_idx.append(dset.all_pair2idx[current])
            self.test_idx = torch.LongTensor(test_idx).to(self.device)
            self.test_idx = remove_from_B(self.train_idx,self.test_idx)
            self.unseen_idx = union_of_AB(self.val_idx, self.test_idx)


        self.Py = _Py(self.dset,self.args)
        self.VsVo = _Vs_Vo(self.dset,self.args)
        self.PsPo = _Ps_Po(self.dset,self.args)

        self.obj2pairs = self.create_obj_pairs().to(self.device)
        self.attr2pairs = self.create_attr_pairs().to(self.device)

        if args.train_only:
            self.obj2pairs = self.obj2pairs[:,self.train_idx].to(self.device)
            self.attr2pairs = self.attr2pairs[:,self.train_idx].to(self.device)

        self.prob = dict()
        self._init_label()

    def _init_label(self):
        self.label_smooth = torch.zeros(self.num_pairs, self.num_pairs)
        for i in range(self.num_pairs):
            for j in range(self.num_pairs):
                if self.pairs[j][1] == self.pairs[i][1]:
                    self.label_smooth[i, j] = self.label_smooth[i, j] + 2
                if self.pairs[j][0] == self.pairs[i][0]:
                    self.label_smooth[i, j] = self.label_smooth[i, j] + 1
        self.label_smooth = self.label_smooth[:, self.train_idx]
        self.label_smooth = self.label_smooth[self.train_idx, :]
        K_1 = (self.label_smooth == 1).sum(dim=1)
        K_2 = (self.label_smooth == 2).sum(dim=1)
        K = K_1 + 2 * K_2
        self.epi = self.args.smooth
        template = torch.ones_like(self.label_smooth) / K
        template = template * self.epi
        self.label_smooth = self.label_smooth * template
        for i in range(self.label_smooth.shape[0]):
            self.label_smooth[i, i] = 1 - (self.epi)
        self.label_smooth = self.label_smooth.to(self.device)

    def cross_entropy(self, logits, label):
        logits = F.log_softmax(logits, dim=-1)
        loss = -(logits * label).sum(-1).mean()
        return loss
    def create_obj_pairs(self):
        obj_matrix = torch.zeros(self.num_objs,self.num_pairs)
        for i in range(self.num_objs):
            for j in range(self.num_pairs):
                if self.dset.objs[i] == self.pairs[j][1]:
                    obj_matrix[i,j] = 1
        return obj_matrix
    def create_attr_pairs(self):
        obj_matrix = torch.zeros(self.num_attrs,self.num_pairs)
        for i in range(self.num_attrs):
            for j in range(self.num_pairs):
                if self.dset.attrs[i] == self.pairs[j][0]:
                    obj_matrix[i,j] = 1
        return obj_matrix

    def Classifiers(self, type, feat, attr, objs):
        if type == 'y':
            pyy = self.Py(attr, objs).permute(1, 0)
            logits = torch.mm(feat, pyy)
        elif type == 'so':
            logits = {}
            xs, xo = self.VsVo(feat)
            pss, poo = self.PsPo(attr,objs)
            logits['s'] = torch.mm(xs, pss.permute(1, 0))
            logits['o'] = torch.mm(xo, poo.permute(1, 0))
        else:
            raise ValueError("Please enter the correct classifier type")
        return logits

    def _init_cliques(self):
        train_pairs = [self.pairs[i] for i in self.train_idx]
        cliques = torch.eye(len(self.train_idx)) * 5
        for i in range(len(self.train_idx)):
            for j in range(len(self.train_idx)):
                # obj
                if train_pairs[i][1] == train_pairs[j][1] and i != j:
                    idx = self.dset.obj2idx[train_pairs[j][1]]
                    if self.prob['objs_y'][i] < self.prob['objs'][idx]:
                        cliques[i, j] = 3
                    else:
                        cliques[i, j] = 4
                # adj
                if train_pairs[i][0] == train_pairs[j][0] and i != j:
                    idx = self.dset.attr2idx[train_pairs[j][0]]
                    if self.prob['attr_y'][i] < self.prob['attr'][idx]:
                        cliques[i, j] = 1
                    else:
                        cliques[i, j] = 2
        self.cliques = cliques

    def calc_loss(self, y_true, y_pred):
        y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
        # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
        y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
        y_true = y_true.float()
        y_pred = y_pred - (1 - y_true) * 1e12
        y_pred = y_pred.view(-1)
        if torch.cuda.is_available():
            y_pred = torch.cat((torch.tensor([0]).float().to(self.device), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        else:
            y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1

        return torch.logsumexp(y_pred, dim=0)

    def get_x(self, img):
        xy, xso = None, None
        if self.args.nlayers:
            xy = self.image_embedder(img)
        xso = img
        if self.args.P_y == 'MLP':
            xy = F.normalize(xy, dim=1)
        xso = F.normalize(xso, dim=-1)
        return xy, xso
    def freeze_model(self):
        for p in self.Py.parameters():
            p.requires_grad = False
        for p in self.image_embedder.parameters():
            p.requires_grad = False
    def unfreeze_model(self):
        for p in self.Py.parameters():
            p.requires_grad = True
        for p in self.image_embedder.parameters():
            p.requires_grad = True

    def val_forward(self, x):
        img = x[0]
        img = F.avg_pool2d(img, kernel_size=7).view(-1, self.dset.feat_dim)
        feats_y, feats_so = self.get_x(img)

        pred_y = self.Classifiers('y', feats_y, self.val_attrs,self.val_objs)
        pred_so = self.Classifiers('so', feats_so, self.val_attrs,self.val_objs)
        pred_so['s'] = F.softmax(pred_so['s'], dim=-1)
        pred_so['o'] = F.softmax(pred_so['o'], dim=-1)

        s = pred_y + self.args.eta * torch.log((pred_so['s'] * pred_so['o']))
        score = F.softmax(s, dim=-1)
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]
        return None, scores

    def train_forward_closed(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        img = F.avg_pool2d(img, kernel_size=7).view(-1, self.dset.feat_dim)
        smoothed_labels = self.label_smooth[pairs]

        feats_y, feats_so = self.get_x(img)

        pred_y = self.Classifiers('y', feats_y, self.train_attrs, self.train_objs)
        L_cls = self.cross_entropy(self.scale * (pred_y), smoothed_labels)

        pred_so = self.Classifiers('so', feats_so, self.uniq_attrs,self.uniq_objs)
        L_s = F.cross_entropy(self.scale * pred_so['s'], attrs)
        L_o = F.cross_entropy(self.scale * pred_so['o'], objs)

        clique_label = self.cliques[pairs,:].to(self.device)
        loss_clique = self.calc_loss(clique_label, self.scale * pred_y)
        loss = L_cls + L_s + L_o + self.args.alpha * loss_clique
        return loss

    def train_forward_prior(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        img = F.avg_pool2d(img, kernel_size = 7).view(-1, self.dset.feat_dim)
        feats_y, feats_so = self.get_x(img)
        pred_so = self.Classifiers('so', feats_so, self.uniq_attrs,self.uniq_objs)
        L_s = F.cross_entropy(self.scale * pred_so['s'], attrs)
        L_o = F.cross_entropy(self.scale * pred_so['o'], objs)
        loss = L_s + L_o
        with torch.no_grad():
            poster = {}
            p_yx = []
            prob_s = F.softmax(pred_so['s'],dim=-1).detach()
            prob_o = F.softmax(pred_so['o'], dim=-1).detach()
            p_yx.append(prob_s@self.attr2pairs)
            p_yx.append(prob_o@self.obj2pairs)
            mask_a = torch.zeros_like(p_yx[0])
            mask_o = torch.zeros_like(p_yx[1])
            mask_a[:, pairs] = 1
            mask_o[:, pairs] = 1
            p_yx[0] = p_yx[0] * mask_a
            p_yx[1] = p_yx[1] * mask_o
            p_sox = []
            p_sox.append(prob_s)
            p_sox.append(prob_o)

            mask_a = torch.zeros_like(p_sox[0])
            mask_o = torch.zeros_like(p_sox[1])

            mask_a[:, attrs] = 1
            mask_o[:, objs] = 1

            p_sox[0] = p_sox[0] * mask_a
            p_sox[1] = p_sox[1] * mask_o

            poster['p_yx'] = p_yx
            poster['p_cx'] = p_sox
        return loss, poster

    def forward(self, x, prior = False):
        if self.training:
            priors = None
            if prior == False:
                out = self.train_forward_closed(x)
            else:
                out, priors = self.train_forward_prior(x)

            return out, priors
        else:
            with torch.no_grad():
                _, out = self.val_forward(x)
            return out




