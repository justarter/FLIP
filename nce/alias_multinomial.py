from math import isclose
import pickle as pkl
import json
import os

import torch

class AliasMultinomial(torch.nn.Module):
    '''Alias sampling method to speedup multinomial sampling

    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the save multinomial distribution.

    Attributes:
        - probs: the probability density of desired multinomial distribution

    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs, data_path):
        super(AliasMultinomial, self).__init__()
        sum_probs = int(probs.sum().item() * 100000) / 100000.0
        # assert isclose(sum_probs, 1), 'The noise distribution must sum to 1, but get {}'.format(sum_probs)
        cpu_probs = probs.cpu()
        K = len(probs)

        self_prob_file = data_path + f'alias_self_prob-core.h5'
        self_alias_file = data_path + f'alias_self_alias-core.h5'
        if os.path.exists(self_prob_file) and os.path.exists(self_alias_file):
            self_prob = torch.load(self_prob_file)
            self_alias = torch.load(self_alias_file)
        else:
            # such a name helps to avoid the namespace check for nn.Module
            self_prob = [0] * K
            self_alias = [0] * K

            # Sort the data into the outcomes with probabilities
            # that are larger and smaller than 1/K.
            smaller = []
            larger = []
            for idx, prob in enumerate(cpu_probs):
                self_prob[idx] = K*prob
                if self_prob[idx] < 1.0:
                    smaller.append(idx)
                else:
                    larger.append(idx)

            # Loop though and create little binary mixtures that
            # appropriately allocate the larger outcomes over the
            # overall uniform mixture.
            while len(smaller) > 0 and len(larger) > 0:
                small = smaller.pop()
                large = larger.pop()

                self_alias[small] = large
                self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

                if self_prob[large] < 1.0:
                    smaller.append(large)
                else:
                    larger.append(large)

            for last_one in smaller + larger:
                self_prob[last_one] = 1

            self_prob = torch.Tensor(self_prob)
            self_alias = torch.LongTensor(self_alias)
            
            torch.save(self_prob, self_prob_file)
            torch.save(self_alias, self_alias_file)

        self.register_buffer('prob', self_prob)
        self.register_buffer('alias', self_alias)

    def draw(self, *size):
        """Draw N samples from multinomial

        Args:
            - size: the output size of samples
        """
        max_value = self.alias.size(0)

        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)

        return (oq + oj).view(size)

