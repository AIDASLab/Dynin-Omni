# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def grad_multiply_wrapper(module, rate):
    class GMModel(torch.nn.Module):
        def __init__(self):
            super(GMModel, self).__init__()
            self.module = module
            self.rate = rate

        def forward(self, *args, **kwargs):
            if self.rate > 0:
                features = self.module(*args, **kwargs)
                if self.rate != 1.0:
                    if isinstance(features, torch.Tensor):
                        features = GradMultiply.apply(features, self.rate)
                    elif isinstance(features, tuple):
                        features = (GradMultiply.apply(f, self.rate) for f in features)
                    elif isinstance(features, dict):
                        features = {k: GradMultiply.apply(f, self.rate) for k, f in features.items()}
            else:
                with torch.no_grad():
                    features = module(*args, **kwargs)
            return features

    return GMModel()


if __name__ == '__main__':
    class M(torch.nn.Module):
        def __init__(self):
            super(M, self).__init__()
            self.a = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return self.a * x


    m = M()
    m = grad_multiply_wrapper(m, 0.3)
    optim = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0)
    optim.zero_grad()
    loss = m(torch.ones(1))
    loss.backward()
    optim.step()
    print(m.module.a)
