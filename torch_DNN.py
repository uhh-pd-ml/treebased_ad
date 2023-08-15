from torch import nn


class DNNClassifier(nn.Module):
    def __init__(self, input_dim, layers):
        super().__init__()
        self.input_dim = input_dim

        self.layers = []

        for idx, nodes in enumerate(layers):
            if idx == 0:
                self.layers.append(nn.Linear(self.input_dim, nodes))
                self.layers.append(nn.ReLU())
            elif idx == len(layers) - 1:
                self.layers.append(nn.Linear(layers[idx-1], nodes))
                self.layers.append(nn.Sigmoid())
            else:
                self.layers.append(nn.Linear(layers[idx-1], nodes))
                self.layers.append(nn.ReLU())

        self.model_stack = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model_stack(x)
