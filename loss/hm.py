import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd

from torch.cuda.amp import custom_fwd, custom_bwd

class myHM(autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs):
        ctx.inputs = inputs
        outputs = inputs.mm(ctx.inputs.t())

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.inputs)

        return grad_inputs, None, None, None

def myhm(inputs):
    return myHM.apply(inputs)

class myHybridMemory(nn.Module):
    def __init__(self, interval, num_players, num_instances, num_actions, temp=0.05):
        super(myHybridMemory, self).__init__()
        self.interval = interval
        self.num_players = num_players
        self.num_instances = num_instances
        self.num_actions = num_actions
        self.temp = temp

        # self.labels = torch.arange(0,self.num_players).view(self.num_players,1).expand(self.num_players,self.num_instances).reshape(-1).cuda()
        self.labels = torch.arange(0,self.num_players * self.num_actions).view(self.num_players * self.num_actions,1).expand(self.num_players * self.num_actions,self.num_instances).reshape(-1).cuda()
        print(self.labels)
        print('contrast loss interval:', self.interval, 'num players:', self.num_players)

    def forward(self, inputs):
        inputs = F.normalize(inputs, p=2, dim=1)

        # B1, D = inputs.size()
        # inputs = inputs.view(-1, self.interval, D)
        # B2 = inputs.size(0)
        # loss=torch.tensor(0.).cuda()
        # for B_idx in range(B2):
        #     _input = inputs[B_idx]
        #     # inputs: B*2048, features: L*2048
        #     _input = myhm(_input)
        #     _input /= self.temp
        #     B = _input.size(0)
        #
        #     targets = self.labels.clone()
        #     sim = _input.view(B,-1,self.num_instances).mean(2)
        #     sim = torch.exp(sim)
        #     sim_sums = sim.sum(1, keepdim=True) + 1e-6
        #     sim = sim/sim_sums
        #     _loss = F.nll_loss(torch.log(sim+1e-6), targets)
        #     loss += _loss
        # loss = loss / B2
        # return loss

        B = inputs.size(0)
        # inputs: B*2048, features: L*2048
        inputs = myhm(inputs)
        inputs /= self.temp
        B = inputs.size(0)

        targets = self.labels.clone()
        sim = inputs.view(B,-1,self.num_instances).mean(2)
        sim = torch.exp(sim)
        sim_sums = sim.sum(1, keepdim=True) + 1e-6
        sim = sim/sim_sums
        return F.nll_loss(torch.log(sim+1e-6), targets)
