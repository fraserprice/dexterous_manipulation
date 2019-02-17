import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

'''
Code for the implementation of Curiosity-driven Exploration by Self-supervised Prediction.
This allows us to build a model of states we should "curiously" explore. It contains:
 
Forward model: enc(s_t), a --> enc(s_hat_t+1)
    The output of this (s_hat_t+1) can be used to measure how well the agent is able to determine the
    consequences of its actions when combined with ground truth s_t+1. This can give us an idea of which states the
    agent has not yet explored, and so MSE(s_t+1, s_hat_t+1) can be used as an intrinsic reward signal.
    
Inverse model: s_t, s_t+1 --> enc(s_t), enc(s_t+1) --> a_hat
    This model is used to extract information from our state which is relevant to our agent, i.e:
        1. Things the agent can control
        2. Things the agent cannot control but affect the agent
    This has effectively discards information which the agent cannot control and does not affect it- it does so by
    creating a state embedding enc(s_t).
    This step is important because otherwise our agent may become stuck being "curious" about things which do not
    affect it.
    It is worth noting that this step is not necessary for our current use case, as all elements of the state are
    relevant to the agent. It is useful in scenarios where the state may be high-dimensional (e.g. images) or may
    contain superfluous information.
    
'''


# noinspection PyArgumentList
class CuriosityModule:
    def __init__(self, action_space_size, state_space_size, forward_hidden, use_backward_dynamics=False, dropout=0.4,
                 examples_before_optimization=1000):
        self.action_space = action_space_size
        self.state_space = state_space_size
        self.forward_dynamics = None
        self.backward_dynamics = None
        self.initialize_forward_dynamics(forward_hidden, dropout=dropout)
        self.use_backward_dynamics = use_backward_dynamics
        if use_backward_dynamics:
            self.initialize_backward_dynamics()
        self.xs = []
        self.ys = []
        self.examples_before_optimization = examples_before_optimization

    def get_curiosity_loss(self, action, s_t_enc, s_t_plus_1_enc, optim=True):
        x = np.concatenate((action, s_t_enc))
        if optim:
            self.xs.append(x)
            self.ys.append(s_t_plus_1_enc)
            if len(self.xs) > self.examples_before_optimization:
                print("Optimizing curiosity")
                self.optimize_forward_dynamics(self.xs, self.ys)
                print("Finished curiosity optimization")
                self.xs = []
                self.ys = []
        s_hat_t_plus_1_enc = self.forward_dynamics(Variable(torch.Tensor([x])))
        s_t_plus_1_enc = Variable(torch.Tensor([s_t_plus_1_enc]))
        return torch.sum((s_hat_t_plus_1_enc - s_t_plus_1_enc) ** 2)

    def encode_state(self, state):
        if self.use_backward_dynamics:
            raise NotImplementedError("Not yet implemented")
        return state

    def save_forward(self, path):
        torch.save(self.forward_dynamics.state_dict(), path)

    def load_forward(self, path):
        print("Loading!")
        self.forward_dynamics.load_state_dict(torch.load(path))
        self.forward_dynamics.eval()

    def initialize_forward_dynamics(self, hidden_sizes, dropout):
        self.forward_dynamics = ForwardDynamics(self.action_space, hidden_sizes, self.state_space, dropout=dropout)

    def initialize_backward_dynamics(self):
        self.backward_dynamics = BackwardDynamics()

    def optimize_forward_dynamics(self, xs, ys, lr=0.00001, weight_decay=0.0001, batch_size=16):
        cost = nn.MSELoss()
        optimizer = Adam(self.forward_dynamics.parameters(), lr=lr, weight_decay=weight_decay)
        nb_epochs = 10
        xs_loader = DataLoader(TensorDataset(torch.Tensor(xs)), batch_size=batch_size)
        ys_loader = DataLoader(TensorDataset(torch.Tensor(ys)), batch_size=batch_size)
        for epoch in range(nb_epochs):
            ys_iter = iter(ys_loader)
            for i, x_batch in enumerate(xs_loader):
                y_pred = self.forward_dynamics(x_batch)
                loss = cost(y_pred, Variable(torch.Tensor(next(ys_iter)[0])))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # # xs_val_unk = [(random.random(), random.random() * 2) for _ in range(100)]
            # # ys_val_unk = [(a + b) for (a, b) in xs_val_unk]
            # xs_val = [(random.random(), random.random()) for _ in range(100)]
            # ys_val = [(a + b) for (a, b) in xs_val]
            # cur_losses = [self.get_curiosity_loss([xs_val[i][1]], [xs_val[i][0]], y) for i, y in enumerate(ys_val)]
            # # cur_losses_unk = [self.get_curiosity_loss([xs_val_unk[i][1]], [xs_val_unk[i][0]], y) for i, y in enumerate(ys_val_unk)]
            # print(f"Curiosity loss known a: {sum(cur_losses) / len(cur_losses)},\n ")
            #       # f"Curiosity loss unknown a: {sum(cur_losses_unk) / len(cur_losses_unk)}")

    def optimize_backward_dynamics(self):
        raise NotImplementedError("Not yet implemented")


class ForwardDynamics(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.4):
        super(ForwardDynamics, self).__init__()
        self.layers = []
        self.params = nn.ParameterList()
        self.__init(input_size, hidden_sizes, output_size, dropout)

    def __init(self, input_size, hidden_sizes, output_size, dropout):
        current_size = input_size
        for i, layer_size in enumerate(hidden_sizes):
            layer = nn.Linear(current_size, layer_size)
            self.layers.append(layer)
            self.params.append(layer.weight)
            self.params.append(layer.bias)
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            current_size = layer_size
        self.layers.append(nn.Linear(current_size, output_size))
        # self.layers.append(nn.Sigmoid())

    def forward(self, x):
        out = x
        for layer in self.layers:
            if isinstance(out, list):
                out = layer(out[0])
            else:
                out = layer(out)
        return out


class BackwardDynamics(nn.Module):
    def __init__(self):
        super(BackwardDynamics, self).__init__()
        self.feature_extraction_layers = []
        self.prediction_layers = []
        self.params = nn.ParameterList()
        self.__init()

    def __init(self):
        raise NotImplementedError("Not yet implemented")

    def forward(self, x):
        raise NotImplementedError("Not yet implemented")


# cm = CuriosityModule(2, 1, [32, 32, 32, 32], dropout=0.5)
# examples = 1000
# xs = [(random.random(), random.random()) for _ in range(examples)]
# ys = [(a + b) for (a, b) in xs]
# cm.optimize_forward_dynamics(xs, ys, batch_size=1)
