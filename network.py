import torch.nn as nn

def mlp(nodes, activations):
    layer = []

    for i, act in enumerate(activations):
        layer += [nn.Linear(nodes[i], nodes[i+1]), act()]
    
    return nn.Sequential(*layer) 

if __name__=='__main__':
    net = mlp([3, 4], [nn.Tanh])

    import torch 
    x = torch.tensor([[2.0, 5.0, 6.0]])

    z = net(x).sum()
    z.backward()

    W, b = [param for param in net.parameters()]

    print(z)
    print(W.shape, b.shape)
    print(W@x.T+b.T)
    dzdy = 1 / torch.cosh(W@x.T+b)**2 
    print(dzdy.T)
    dzdW = dzdy @ x
    print(dzdW)


