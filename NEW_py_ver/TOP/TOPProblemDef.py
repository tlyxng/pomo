
import torch
import numpy as np


def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    #shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    #shape: (batch, problem, 2)

    # uniform distribution for prize first
    # shape： (batch, problem)
    node_prize = (1 + torch.randint(0, 100, size=(batch_size, problem_size))).float() / 100

    # Fischetti et al. recommended max_length (half of expected optimal TSP tour)
    default_max_length = {20: 2.0, 50: 3.0, 100: 4.0}.get(problem_size, 3.0)
    max_length = torch.full((batch_size, 1), default_max_length)

    return depot_xy, node_xy, node_prize, max_length

def augment_xy_data_by_8_fold(xy_data):
    # problems.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_xy_data