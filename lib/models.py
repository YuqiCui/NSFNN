import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as fuzz


def kmean_init(x_train, n_rules, scale=10., m=None):  # m for unify the input arguments
    km = KMeans(n_rules, n_init=1)
    km.fit(x_train)
    Cs = km.cluster_centers_.T
    yc = km.predict(x_train)

    vs = np.zeros([x_train.shape[1], n_rules])
    dists = np.zeros([n_rules, x_train.shape[1]])
    for i in range(n_rules):
        dists[i] = np.std((x_train[yc == i, :] - km.cluster_centers_[i, :]), axis=0)  # Dongrui Wu
        vs[:, i] = np.maximum(dists[i], 1e-8).T * scale
    return Cs, vs


def fcm_init(x_train, n_rules, m=None, scale=1.):
    if m is not None:
        assert m > 1, "m must be larger than 1, received: {}".format(m)
    else:
        if min(x_train.shape[0], x_train.shape[1] - 1) >= 3:
            m = min(x_train.shape[0], x_train.shape[1] - 1) / (min(x_train.shape[0], x_train.shape[1] - 1) - 2)
        else:
            m = 2
    n_samples, n_features = x_train.shape
    centers, mem, _, _, _, _, _ = fuzz.cmeans(
        x_train.T, n_rules, m, error=1e-5, maxiter=200)
    delta = np.zeros([n_rules, n_features])
    for i in range(n_rules):
        d = (x_train - centers[i, :]) ** 2
        delta[i, :] = np.sum(d * mem[i, :].reshape(-1, 1), axis=0) / np.sum(mem[i, :])
    delta = np.sqrt(delta) * scale
    delta = np.where(delta < 0.05, 0.05, delta)
    return centers.T, delta.T


class FuzzyNeuralNework(nn.Module):
    def __init__(self, in_dim, n_rules, n_classes,
                 init_centers=None, init_sigmas=None,
                 weight_path=None, ampli=0, defuzzy='norm', ravel_out=False):
        """
        init
        :param in_dim: int, number of features
        :param n_rules: int. number of rules
        :param n_classes: int, number of classes
        :param init_centers: np.darray, init center vector, [in_dim, n_rules]
        :param init_sigmas: np.darray, init standard deviation for fuzzy sets, [in_dim, n_rules]
        :param ampli: int, add one term to the exp to avoid numeric underflow
        :param defuzzy: 'norm' or 'sum'.

        """
        super(FuzzyNeuralNework, self).__init__()
        self.in_dim = in_dim
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.init_centers = init_centers
        self.init_sigmas = init_sigmas
        self.weight_path = weight_path
        self.ampli = ampli
        self.defuzzy_method = defuzzy
        self.bn1 = nn.BatchNorm1d(num_features=self.in_dim)
        self.ravel_out = ravel_out

        if self.weight_path is None:
            assert self.init_centers.shape == (self.in_dim, self.n_rules), \
                "Shape of init_centers must be {}".format((self.in_dim, self.n_rules))
            assert self.init_sigmas.shape == (self.in_dim, self.n_rules), \
                "Shape of init_sigmas must be {}".format((self.in_dim, self.n_rules))

        self.build_model()
        self.masked_param_name = ['centers', 'sigmas', 'weights']  # do not mask BN layer and biases

    def build_model(self):
        self.eps = 1e-8

        self.weights = torch.zeros(size=(self.n_rules, self.in_dim, self.n_classes))
        self.biases = torch.zeros(size=(1, self.n_rules, self.n_classes))
        self.centers = torch.zeros(size=(self.in_dim, self.n_rules))
        self.sigmas = torch.zeros(size=self.centers.size())

        self.weights = nn.Parameter(self.weights, requires_grad=True)
        self.biases = nn.Parameter(self.biases, requires_grad=True)
        self.centers = nn.Parameter(self.centers, requires_grad=True)
        self.sigmas = nn.Parameter(self.sigmas, requires_grad=True)

        if self.defuzzy_method == 'norm':
            self.defuzzy = lambda frs, dim=-1: frs / (torch.sum(frs, dim=dim, keepdim=True) + 1e-10)
        elif self.defuzzy_method == 'sum':
            self.defuzzy = lambda frs: frs
        else:
            raise ValueError("unsupported defuzzy method {}".format(self.defuzzy_method))

        self.rule_masks = torch.ones(self.n_rules, requires_grad=False)

        if self.weight_path is not None:
            self.load_state_dict(torch.load(self.weight_path))
            return

        self.centers.data = torch.as_tensor(self.init_centers).float()
        self.sigmas.data = torch.as_tensor(self.init_sigmas).float()

        nn.init.normal_(self.weights, std=1/math.sqrt(self.in_dim))
        nn.init.constant_(self.biases, 0)

    def forward(self, x):
        raw_frs = torch.exp(
            torch.sum(
                -torch.pow(x.unsqueeze(dim=2) - self.centers, 2) / (2 * torch.pow(self.sigmas, 2)), dim=1
            ) + self.ampli
        )
        raw_frs = raw_frs * self.rule_masks  # convenient for rule dropping experiment
        frs = self.defuzzy(raw_frs)
        x = self.bn1(x)
        x_rep = x.unsqueeze(dim=1).expand([x.size(0), self.n_rules, x.size(1)])
        cons = torch.einsum('ijk,jkl->ijl', [x_rep, self.weights])
        cons = cons + self.biases
        outs = torch.mul(cons, frs.unsqueeze(2))
        return torch.sum(outs, dim=1, keepdim=False).view(-1) if self.ravel_out else torch.sum(outs, dim=1, keepdim=False)

    def l2_loss(self):
        """
        compute the l2 loss using the consequent parameters except the bias
        :return:
        """
        return torch.sum(self.weights ** 2)

    def set_rule_masks(self, masks):
        self.rule_masks.data[...] = masks


class NonSingletonFNN(nn.Module):
    def __init__(self, in_dim, n_rules, n_classes,
                 init_centers=None, init_sigmas=None,
                 weight_path=None, ampli=0, defuzzy='norm', init_ns_sigma=1., ravel_out=False):
        """
        init
        :param in_dim: int, number of features
        :param n_rules: int. number of rules
        :param n_classes: int, number of classes
        :param init_centers: np.darray, init center vector, [in_dim, n_rules]
        :param init_sigmas: np.darray, init standard deviation for fuzzy sets, [in_dim, n_rules]
        :param ampli: int, add one term to the exp to avoid numeric underflow
        :param defuzzy: 'norm' or 'sum'.
        :param init_ns_sigma: int or array. int: sigma same for all attributes, else, size of [in_dim, ] array
        """
        super(NonSingletonFNN, self).__init__()
        self.in_dim = in_dim
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.init_centers = init_centers
        self.init_sigmas = init_sigmas
        self.weight_path = weight_path
        self.ampli = ampli
        self.defuzzy_method = defuzzy
        self.bn1 = nn.BatchNorm1d(num_features=self.in_dim)
        self.ravel_out = ravel_out

        if isinstance(init_ns_sigma, float):
            self.init_ns_sigma = np.ones([self.in_dim]) * init_ns_sigma
        else:
            self.init_ns_sigma = np.array(init_ns_sigma)

        if self.weight_path is None and self.init_centers is not None and \
                self.init_sigmas is not None and self.init_ns_sigma is not None:
            assert self.init_centers.shape == (self.in_dim, self.n_rules), \
                "Shape of init_centers must be {}".format((self.in_dim, self.n_rules))
            assert self.init_sigmas.shape == (self.in_dim, self.n_rules), \
                "Shape of init_sigmas must be {}".format((self.in_dim, self.n_rules))
            assert self.init_ns_sigma.shape[0] == self.in_dim, \
                "init_ns_sigma shape is wrong, got {}".format(self.init_ns_sigma.shape)

        self.build_model()
        self.masked_param_name = ['centers', 'sigmas', 'ns_sigma', 'weights']  # do not mask BN layer and biases

    def build_model(self):
        self.eps = 1e-8

        self.weights = torch.zeros(size=(self.n_rules, self.in_dim, self.n_classes))
        self.biases = torch.zeros(size=(1, self.n_rules, self.n_classes))
        self.centers = torch.zeros(size=(self.in_dim, self.n_rules))
        self.sigmas = torch.zeros(size=self.centers.size())
        self.ns_sigma = torch.zeros(size=(self.in_dim, 1))

        self.weights = nn.Parameter(self.weights, requires_grad=True)
        self.biases = nn.Parameter(self.biases, requires_grad=True)
        self.centers = nn.Parameter(self.centers, requires_grad=True)
        self.sigmas = nn.Parameter(self.sigmas, requires_grad=True)
        self.ns_sigma = nn.Parameter(self.ns_sigma, requires_grad=True)

        if self.defuzzy_method == 'norm':
            self.defuzzy = lambda frs, dim=-1: frs / (torch.sum(frs, dim=dim, keepdim=True) + 1e-10)
        elif self.defuzzy_method == 'sum':
            self.defuzzy = lambda frs: frs
        else:
            raise ValueError("unsupported defuzzy method {}".format(self.defuzzy_method))

        self.rule_masks = torch.ones(self.n_rules, requires_grad=False)

        if self.weight_path is not None:
            self.load_state_dict(torch.load(self.weight_path))
            return

        if self.init_centers is not None:
            self.centers.data[...] = torch.as_tensor(self.init_centers).float()
        if self.init_sigmas is not None:
            self.sigmas.data[...] = torch.as_tensor(self.init_sigmas).float()
        if self.init_ns_sigma is not None:
            self.ns_sigma.data[...] = torch.as_tensor(self.init_ns_sigma).float().view(-1, 1)

        nn.init.normal_(self.weights, std=1/math.sqrt(self.in_dim))
        nn.init.constant_(self.biases, 0)

    def forward(self, x):
        raw_frs = torch.exp(
            torch.sum(
                -torch.pow(x.unsqueeze(dim=2) - self.centers, 2) / (2 * torch.pow(self.sigmas, 2) + torch.pow(self.ns_sigma, 2)), dim=1
            ) + self.ampli
        )
        raw_frs = raw_frs * self.rule_masks  # convenient for rule dropping experiment
        frs = self.defuzzy(raw_frs)
        x = self.bn1(x)
        x_rep = x.unsqueeze(dim=1).expand([x.size(0), self.n_rules, x.size(1)])
        cons = torch.einsum('ijk,jkl->ijl', [x_rep, self.weights])
        cons = cons + self.biases
        outs = torch.mul(cons, frs.unsqueeze(2))
        return torch.sum(outs, dim=1, keepdim=False).view(-1) if self.ravel_out else torch.sum(outs, dim=1, keepdim=False)

    def l2_loss(self):
        """
        compute the l2 loss using the consequent parameters except the bias
        :return:
        """
        return torch.sum(self.weights ** 2)

    def set_rule_masks(self, masks):
        self.rule_masks.data[...] = masks


class NNonSingletonFNN(nn.Module):
    def __init__(self, in_dim, n_rules, n_classes,
                 init_centers=None, init_sigmas=None,
                 weight_path=None, ampli=0, defuzzy='norm', init_ns_sigma=1., ravel_out=False):
        """
        init
        :param in_dim: int, number of features
        :param n_rules: int. number of rules
        :param n_classes: int, number of classes
        :param init_centers: np.darray, init center vector, [in_dim, n_rules]
        :param init_sigmas: np.darray, init standard deviation for fuzzy sets, [in_dim, n_rules]
        :param ampli: int, add one term to the exp to avoid numeric underflow
        :param defuzzy: 'norm' or 'sum'.
        :param init_ns_sigma: int or array. int: sigma same for all attributes, else, size of [in_dim, ] array
        """
        super(NNonSingletonFNN, self).__init__()
        self.in_dim = in_dim
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.init_centers = init_centers
        self.init_sigmas = init_sigmas
        self.weight_path = weight_path
        self.ampli = ampli
        self.defuzzy_method = defuzzy
        self.bn1 = nn.BatchNorm1d(num_features=self.in_dim)
        self.ravel_out = ravel_out

        if isinstance(init_ns_sigma, float):
            self.init_ns_sigma = np.ones([self.in_dim, self.n_rules]) * init_ns_sigma
        else:
            self.init_ns_sigma = np.expand_dims(np.array(init_ns_sigma), axis=1).repeat(self.n_rules, axis=1)

        if self.weight_path is None and self.init_centers is not None and \
                self.init_sigmas is not None and self.init_ns_sigma is not None:
            assert self.init_centers.shape == (self.in_dim, self.n_rules), \
                "Shape of init_centers must be {}".format((self.in_dim, self.n_rules))
            assert self.init_sigmas.shape == (self.in_dim, self.n_rules), \
                "Shape of init_sigmas must be {}".format((self.in_dim, self.n_rules))
            assert self.init_ns_sigma.shape[0] == self.in_dim, \
                "init_ns_sigma shape is wrong, got {}".format(self.init_ns_sigma.shape)

        self.build_model()
        self.masked_param_name = ['centers', 'sigmas', 'ns_sigma', 'weights']  # do not mask BN layer and biases

    def build_model(self):
        self.eps = 1e-8

        self.weights = torch.zeros(size=(self.n_rules, self.in_dim, self.n_classes))
        self.biases = torch.zeros(size=(1, self.n_rules, self.n_classes))
        self.centers = torch.zeros(size=(self.in_dim, self.n_rules))
        self.sigmas = torch.zeros(size=self.centers.size())
        self.ns_sigma = torch.zeros(size=(self.in_dim, self.n_rules))

        self.weights = nn.Parameter(self.weights, requires_grad=True)
        self.biases = nn.Parameter(self.biases, requires_grad=True)
        self.centers = nn.Parameter(self.centers, requires_grad=True)
        self.sigmas = nn.Parameter(self.sigmas, requires_grad=True)
        self.ns_sigma = nn.Parameter(self.ns_sigma, requires_grad=True)

        if self.defuzzy_method == 'norm':
            self.defuzzy = lambda frs, dim=-1: frs / (torch.sum(frs, dim=dim, keepdim=True) + 1e-10)
        elif self.defuzzy_method == 'sum':
            self.defuzzy = lambda frs: frs
        else:
            raise ValueError("unsupported defuzzy method {}".format(self.defuzzy_method))

        self.rule_masks = torch.ones(self.n_rules, requires_grad=False)

        if self.weight_path is not None:
            self.load_state_dict(torch.load(self.weight_path))
            return

        if self.init_centers is not None:
            self.centers.data[...] = torch.as_tensor(self.init_centers).float()
        if self.init_sigmas is not None:
            self.sigmas.data[...] = torch.as_tensor(self.init_sigmas).float()
        if self.init_ns_sigma is not None:
            self.ns_sigma.data[...] = torch.as_tensor(self.init_ns_sigma).float()

        nn.init.normal_(self.weights, std=1/math.sqrt(self.in_dim))
        nn.init.constant_(self.biases, 0)

    def forward(self, x):
        raw_frs = torch.exp(
            torch.sum(
                -torch.pow(x.unsqueeze(dim=2) - self.centers, 2) / (2 * torch.pow(self.sigmas, 2) + torch.pow(self.ns_sigma, 2)), dim=1
            ) + self.ampli
        )
        raw_frs = raw_frs * self.rule_masks  # convenient for rule dropping experiment
        frs = self.defuzzy(raw_frs)
        x = self.bn1(x)
        x_rep = x.unsqueeze(dim=1).expand([x.size(0), self.n_rules, x.size(1)])
        cons = torch.einsum('ijk,jkl->ijl', [x_rep, self.weights])
        cons = cons + self.biases
        outs = torch.mul(cons, frs.unsqueeze(2))
        return torch.sum(outs, dim=1, keepdim=False).view(-1) if self.ravel_out else torch.sum(outs, dim=1, keepdim=False)

    def l2_loss(self):
        """
        compute the l2 loss using the consequent parameters except the bias
        :return:
        """
        return torch.sum(self.weights ** 2)

    def set_rule_masks(self, masks):
        self.rule_masks.data[...] = masks

