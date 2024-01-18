from typing import Union

import pykeops.torch as keops
import torch

import tqdm


def sinkhorn(x: torch.Tensor, y: torch.Tensor, p: float = 2,
             w_x: Union[torch.Tensor, None] = None,
             w_y: Union[torch.Tensor, None] = None,
             eps: float = 1e-3,
             max_iters: int = 100, stop_thresh: float = 1e-5,
             verbose=False):
    """
    Compute the Entropy-Regularized p-Wasserstein Distance between two d-dimensional point clouds
    using the Sinkhorn scaling algorithm. This code will use the GPU if you pass in GPU tensors.
    Note that this algorithm can be backpropped through
    (though this may be slow if using many iterations).

    :param x: A [n, d] tensor representing a d-dimensional point cloud with n points (one per row)
    :param y: A [m, d] tensor representing a d-dimensional point cloud with m points (one per row)
    :param p: Which norm to use. Must be an integer greater than 0.
    :param w_x: A [n,] shaped tensor of optional weights for the points x (None for uniform weights). Note that these must sum to the same value as w_y. Default is None.
    :param w_y: A [m,] shaped tensor of optional weights for the points y (None for uniform weights). Note that these must sum to the same value as w_y. Default is None.
    :param eps: The reciprocal of the sinkhorn entropy regularization parameter.
    :param max_iters: The maximum number of Sinkhorn iterations to perform.
    :param stop_thresh: Stop if the maximum change in the parameters is below this amount
    :param verbose: Print iterations
    :return: a triple (d, corrs_x_to_y, corr_y_to_x) where:
      * d is the approximate p-wasserstein distance between point clouds x and y
      * corrs_x_to_y is a [n,]-shaped tensor where corrs_x_to_y[i] is the index of the approximate correspondence in point cloud y of point x[i] (i.e. x[i] and y[corrs_x_to_y[i]] are a corresponding pair)
      * corrs_y_to_x is a [m,]-shaped tensor where corrs_y_to_x[i] is the index of the approximate correspondence in point cloud x of point y[j] (i.e. y[j] and x[corrs_y_to_x[j]] are a corresponding pair)
    """

    if not isinstance(p, int):
        raise TypeError(f"p must be an integer greater than 0, got {p}")
    if p <= 0:
        raise ValueError(f"p must be an integer greater than 0, got {p}")

    if eps <= 0:
        raise ValueError("Entropy regularization term eps must be > 0")

    if not isinstance(p, int):
        raise TypeError(f"max_iters must be an integer > 0, got {max_iters}")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be an integer > 0, got {max_iters}")

    if not isinstance(stop_thresh, float):
        raise TypeError(f"stop_thresh must be a float, got {stop_thresh}")

    if len(x.shape) != 2:
        raise ValueError(f"x must be an [n, d] tensor but got shape {x.shape}")
    if len(y.shape) != 2:
        raise ValueError(f"x must be an [m, d] tensor but got shape {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must match in the last dimension (i.e. x.shape=[n, d], "
                         f"y.shape[m, d]) but got x.shape = {x.shape}, y.shape={y.shape}")

    if w_x is not None:
        if w_y is None:
            raise ValueError("If w_x is not None, w_y must also be not None")
        if len(w_x.shape) > 1:
            w_x = w_x.squeeze()
        if len(w_x.shape) != 1:
            raise ValueError(f"w_x must have shape [n,] or [n, 1] "
                             f"where x.shape = [n, d], but got w_x.shape = {w_x.shape}")
        if w_x.shape[0] != x.shape[0]:
            raise ValueError(f"w_x must match the shape of x in dimension 0 but got "
                             f"x.shape = {x.shape} and w_x.shape = {w_x.shape}")
    if w_y is not None:
        if w_x is None:
            raise ValueError("If w_y is not None, w_x must also be not None")
        if len(w_y.shape) > 1:
            w_y = w_y.squeeze()
        if len(w_y.shape) != 1:
            raise ValueError(f"w_y must have shape [n,] or [n, 1] "
                             f"where x.shape = [n, d], but got w_y.shape = {w_y.shape}")
        if w_x.shape[0] != x.shape[0]:
            raise ValueError(f"w_y must match the shape of y in dimension 0 but got "
                             f"y.shape = {y.shape} and w_y.shape = {w_y.shape}")

    # Distance matrix [n, m]
    x_i = keops.Vi(x)  # [n, 1, d]
    y_j = keops.Vj(y)  # [i, m, d]
    if p == 1:
        M_ij = ((x_i - y_j) ** p).abs().sum(dim=2)  # [n, m]
    else:
        M_ij = ((x_i - y_j) ** p).sum(dim=2) ** (1.0 / p)  # [n, m]

    # Weights [n,] and [m,]
    if w_x is None and w_y is None:
        w_x = torch.ones(x.shape[0]).to(x) / x.shape[0]
        w_y = torch.ones(y.shape[0]).to(x) / y.shape[0]
        w_y *= (w_x.shape[0] / w_y.shape[0])

    sum_w_x = w_x.sum().item()
    sum_w_y = w_y.sum().item()
    if abs(sum_w_x - sum_w_y) > 1e-5:
        raise ValueError(f"Weights w_x and w_y do not sum to the same value, "
                         f"got w_x.sum() = {sum_w_x} and w_y.sum() = {sum_w_y} "
                         f"(absolute difference = {abs(sum_w_x - sum_w_y)}")

    log_a = torch.log(w_x)  # [n]
    log_b = torch.log(w_y)  # [m]

    # Initialize the iteration with the change of variable
    u = torch.zeros_like(w_x)
    v = eps * torch.log(w_y)

    u_i = keops.Vi(u.unsqueeze(-1))
    v_j = keops.Vj(v.unsqueeze(-1))

    if verbose:
        pbar = tqdm.trange(max_iters)
    else:
        pbar = range(max_iters)

    for _ in pbar:
        u_prev = u
        v_prev = v

        summand_u = (-M_ij + v_j) / eps
        u = eps * (log_a - summand_u.logsumexp(dim=1).squeeze())
        u_i = keops.Vi(u.unsqueeze(-1))

        summand_v = (-M_ij + u_i) / eps
        v = eps * (log_b - summand_v.logsumexp(dim=0).squeeze())
        v_j = keops.Vj(v.unsqueeze(-1))

        max_err_u = torch.max(torch.abs(u_prev - u))
        max_err_v = torch.max(torch.abs(v_prev - v))
        if verbose:
            pbar.set_postfix({"Current Max Error": max(max_err_u, max_err_v).item()})
        if max_err_u < stop_thresh and max_err_v < stop_thresh:
            break

    P_ij = ((-M_ij + u_i + v_j) / eps).exp()

    approx_corr_1 = P_ij.argmax(dim=1).squeeze(-1)
    approx_corr_2 = P_ij.argmax(dim=0).squeeze(-1)

    if u.shape[0] > v.shape[0]:
        distance = (P_ij * M_ij).sum(dim=1).sum()
    else:
        distance = (P_ij * M_ij).sum(dim=0).sum()
    return distance, approx_corr_1, approx_corr_2
