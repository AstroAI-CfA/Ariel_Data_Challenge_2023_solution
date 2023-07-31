"""
Defines models via the CDF: a neural network is predicing the parameters of a function fitting the target CDF.
x
"""

import torch


def sum_sigm(X, a, b, c):
    dim = -3

    a = (a / (a.sum(dim=-1, keepdim=True) + 1e-12)).unsqueeze(dim=dim)
    b = b.unsqueeze(dim=dim)
    c = c.unsqueeze(dim=dim)

    return torch.sum(a * torch.sigmoid(b * (X.unsqueeze(dim=-1) - c)), dim=-1)


def ensure_0_1(x):
    return torch.nn.ReLU()(x) - torch.nn.ReLU()(x - 1)


def cdf_sum_of_sigm_and_linear_piecewise(X, theta):
    while len(theta.shape) <= len(X.shape):
        theta = theta.unsqueeze(dim=-2)

    support_left_tail = torch.sigmoid(
        10 * torch.abs(theta[..., 0]) * (X - theta[..., 1])
    )
    support_right_tail = torch.sigmoid(
        10 * torch.abs(theta[..., 2]) * (X - theta[..., 1] - torch.abs(theta[..., 3]))
    )

    piecewise_linear = torch.abs(theta[..., 4]) * (X - theta[..., 1])
    piecewise_linear += (
        torch.abs(theta[..., 6]) - torch.abs(theta[..., 4])
    ) * torch.nn.ReLU()(X - theta[..., 1] - torch.abs(theta[..., 5]))
    # piecewise_linear+=(torch.abs(theta[...,7])-torch.abs(theta[...,6]))*torch.nn.ReLU()(X-theta[...,1]-torch.abs(theta[...,5])-torch.abs(theta[...,8]))
    sum_of_sigmoids = sum_sigm(
        X, torch.abs(theta[..., 9:11]), torch.abs(theta[..., 11:13]), theta[..., 13:15]
    )
    # print(sum_of_sigmoids)

    return (
        (support_left_tail - support_right_tail)
        * ensure_0_1(piecewise_linear + sum_of_sigmoids + theta[..., 15])
        + support_right_tail
    )[0]


def linear_and_cubic_spline_piecewise(X, theta):
    while len(theta.shape) <= len(X.shape):
        theta = theta.unsqueeze(dim=-2)
    # theta=torch.sigmoid(theta)

    x1 = theta[..., 0]
    x0 = -2 * torch.ones_like(x1)
    x3 = x1 + theta[..., 1]
    x2 = theta[..., 2] * (x3 - x1) + x1
    x4 = 3 * torch.ones_like(x1)

    alpha1 = 1 / 2 * (x1 - x0) * theta[..., 3]
    alpha2 = 1 / 2 * (x2 - x1) * theta[..., 4]
    alpha3 = 1 / 2 * (x3 - x2) * theta[..., 5]

    beta1 = 1 / 2 * (x2 - x1) * theta[..., 6]
    beta2 = 1 / 2 * (x3 - x2) * theta[..., 7]
    beta3 = 1 / 2 * (x4 - x3) * theta[..., 8]

    y2 = theta[..., 9]

    lin0 = 0.0 * (X < (x1 - alpha1))
    lin1 = y2 / (x2 - x1) * (X - x1) * (X >= (x1 + beta1)) * (X < (x2 - alpha2))
    lin2 = (
        (1.0 - (1.0 - y2) / (x2 - x3) * (X - x3))
        * (X >= (x2 + beta2))
        * (X < (x3 - alpha3))
    )
    lin3 = 1.0 * (X >= (x3 + beta3))

    lintot = lin0 + lin1 + lin2 + lin3

    Y = torch.cat(
        [
            (0.0 * alpha1).unsqueeze(dim=-1),
            (y2 / (x2 - x1) * beta1).unsqueeze(dim=-1),
            (0.0 * x0).unsqueeze(dim=-1),
            (y2 / (x2 - x1)).unsqueeze(dim=-1),
        ],
        dim=-1,
    )

    M = torch.cat(
        [
            torch.cat(
                [
                    (-1 / 3 * alpha1**3).unsqueeze(dim=-1),
                    (1 / 2 * alpha1**2).unsqueeze(dim=-1),
                    (-alpha1).unsqueeze(dim=-1),
                    (torch.ones_like(alpha1)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
            torch.cat(
                [
                    (1 / 3 * beta1**3).unsqueeze(dim=-1),
                    (1 / 2 * beta1**2).unsqueeze(dim=-1),
                    (beta1).unsqueeze(dim=-1),
                    (torch.ones_like(beta1)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
            torch.cat(
                [
                    (alpha1**2).unsqueeze(dim=-1),
                    (-alpha1).unsqueeze(dim=-1),
                    (torch.ones_like(alpha1)).unsqueeze(dim=-1),
                    (torch.zeros_like(alpha1)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
            torch.cat(
                [
                    (beta1**2).unsqueeze(dim=-1),
                    (beta1).unsqueeze(dim=-1),
                    (torch.ones_like(beta1)).unsqueeze(dim=-1),
                    (torch.zeros_like(beta1)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
        ],
        dim=-2,
    )
    cubic_spline_1_coefficents = torch.linalg.solve(M, Y)
    spl1 = (
        torch.sum(
            cubic_spline_1_coefficents
            * torch.cat(
                [
                    (1 / 3 * (X - x1) ** 3).unsqueeze(dim=-1),
                    (1 / 2 * (X - x1) ** 2).unsqueeze(dim=-1),
                    (X - x1).unsqueeze(dim=-1),
                    (torch.ones_like(X)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ),
            dim=-1,
        )
        * (X >= (x1 - alpha1))
        * (X < (x1 + beta1))
    )

    Y = torch.cat(
        [
            (-y2 / (x2 - x1) * alpha2).unsqueeze(dim=-1),
            ((1 - y2) / (x3 - x2) * beta2).unsqueeze(dim=-1),
            (y2 / (x2 - x1)).unsqueeze(dim=-1),
            ((1 - y2) / (x3 - x2)).unsqueeze(dim=-1),
        ],
        dim=-1,
    )

    M = torch.cat(
        [
            torch.cat(
                [
                    (-1 / 3 * alpha2**3).unsqueeze(dim=-1),
                    (1 / 2 * alpha2**2).unsqueeze(dim=-1),
                    (-alpha2).unsqueeze(dim=-1),
                    (torch.ones_like(alpha2)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
            torch.cat(
                [
                    (1 / 3 * beta2**3).unsqueeze(dim=-1),
                    (1 / 2 * beta2**2).unsqueeze(dim=-1),
                    (beta2).unsqueeze(dim=-1),
                    (torch.ones_like(beta2)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
            torch.cat(
                [
                    (alpha2**2).unsqueeze(dim=-1),
                    (-alpha2).unsqueeze(dim=-1),
                    (torch.ones_like(alpha2)).unsqueeze(dim=-1),
                    (torch.zeros_like(alpha2)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
            torch.cat(
                [
                    (beta2**2).unsqueeze(dim=-1),
                    (beta2).unsqueeze(dim=-1),
                    (torch.ones_like(beta2)).unsqueeze(dim=-1),
                    (torch.zeros_like(beta2)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
        ],
        dim=-2,
    )
    cubic_spline_2_coefficents = torch.linalg.solve(M, Y)
    spl2 = (
        torch.sum(
            cubic_spline_2_coefficents
            * torch.cat(
                [
                    (1 / 3 * (X - x2) ** 3).unsqueeze(dim=-1),
                    (1 / 2 * (X - x2) ** 2).unsqueeze(dim=-1),
                    (X - x2).unsqueeze(dim=-1),
                    (torch.ones_like(X)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ),
            dim=-1,
        )
        * (X >= (x2 - alpha2))
        * (X < (x2 + beta2))
    )

    Y = torch.cat(
        [
            (-(1 - y2) / (x3 - x2) * alpha3).unsqueeze(dim=-1),
            (0.0 * beta3).unsqueeze(dim=-1),
            ((1 - y2) / (x3 - x2)).unsqueeze(dim=-1),
            (0.0 * x3).unsqueeze(dim=-1),
        ],
        dim=-1,
    )

    M = torch.cat(
        [
            torch.cat(
                [
                    (-1 / 3 * alpha3**3).unsqueeze(dim=-1),
                    (1 / 2 * alpha3**2).unsqueeze(dim=-1),
                    (-alpha3).unsqueeze(dim=-1),
                    (torch.ones_like(alpha3)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
            torch.cat(
                [
                    (1 / 3 * beta3**3).unsqueeze(dim=-1),
                    (1 / 2 * beta3**2).unsqueeze(dim=-1),
                    (beta3).unsqueeze(dim=-1),
                    (torch.ones_like(beta3)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
            torch.cat(
                [
                    (alpha3**2).unsqueeze(dim=-1),
                    (-alpha3).unsqueeze(dim=-1),
                    (torch.ones_like(alpha3)).unsqueeze(dim=-1),
                    (torch.zeros_like(alpha3)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
            torch.cat(
                [
                    (beta3**2).unsqueeze(dim=-1),
                    (beta3).unsqueeze(dim=-1),
                    (torch.ones_like(beta3)).unsqueeze(dim=-1),
                    (torch.zeros_like(beta3)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ).unsqueeze(dim=-2),
        ],
        dim=-2,
    )
    cubic_spline_3_coefficents = torch.linalg.solve(M, Y)
    spl3 = (
        torch.sum(
            cubic_spline_3_coefficents
            * torch.cat(
                [
                    (1 / 3 * (X - x3) ** 3).unsqueeze(dim=-1),
                    (1 / 2 * (X - x3) ** 2).unsqueeze(dim=-1),
                    (X - x3).unsqueeze(dim=-1),
                    (torch.ones_like(X)).unsqueeze(dim=-1),
                ],
                dim=-1,
            ),
            dim=-1,
        )
        * (X >= (x3 - alpha3))
        * (X < (x3 + beta3))
    )

    spltot = spl1 + spl2 + spl3

    return lintot + spltot


def monotone_cubic_interpolation(X, theta):
    while len(theta.shape) <= len(X.shape):
        theta = theta.unsqueeze(dim=-2)
    theta = torch.sigmoid(theta)

    x1 = theta[..., 0]
    x0 = -2 * torch.ones_like(x1)
    x3 = x1 + theta[..., 1]
    x2 = theta[..., 2] * (x3 - x1) + x1
    x4 = 3 * torch.ones_like(x1)

    alpha1 = 1 / 2 * (x1 - x0) * theta[..., 3]
    alpha2 = 1 / 2 * (x2 - x1) * theta[..., 4]
    alpha3 = 1 / 2 * (x3 - x2) * theta[..., 5]

    beta1 = 1 / 2 * (x2 - x1) * theta[..., 6]
    beta2 = 1 / 2 * (x3 - x2) * theta[..., 7]
    beta3 = 1 / 2 * (x4 - x3) * theta[..., 8]

    y2 = theta[..., 9]

    xlist = torch.cat(
        [
            (x1 - alpha1).unsqueeze(dim=-1),
            (x1 + beta1).unsqueeze(dim=-1),
            (x2 - alpha2).unsqueeze(dim=-1),
            (x2 + beta2).unsqueeze(dim=-1),
            (x3 - alpha3).unsqueeze(dim=-1),
            (x3 + beta3).unsqueeze(dim=-1),
        ],
        dim=-1,
    )
    zeros = torch.zeros_like(x0.unsqueeze(dim=-1))
    ylist = torch.cat(
        [
            zeros,
            (y2 / (x2 - x1) * beta1).unsqueeze(dim=-1),
            (y2 - y2 / (x2 - x1) * alpha2).unsqueeze(dim=-1),
            (y2 + (1 - y2) / (x3 - x2) * beta2).unsqueeze(dim=-1),
            (1 - (1 - y2) / (x3 - x2) * alpha3).unsqueeze(dim=-1),
            1.0 + zeros,
        ],
        dim=-1,
    )

    def h00(t):
        return 2 * t**3 - 3 * t**2 + 1

    def h10(t):
        return t**3 - 2 * t**2 + t

    def h01(t):
        return -2 * t**3 + 3 * t**2

    def h11(t):
        return t**3 - t**2

    answer = torch.zeros_like(X)
    mlist = torch.zeros_like(xlist)
    alist = torch.zeros_like(xlist)
    blist = torch.zeros_like(xlist)

    mlist[..., 1:-1] = (
        1
        / 2
        * (
            (ylist[..., 2:] - ylist[..., 1:-1]) / (xlist[..., 2:] - xlist[..., 1:-1])
            + (ylist[..., :-2] - ylist[..., 1:-1])
            / (xlist[..., :-2] - xlist[..., 1:-1])
        )
    )

    for k in range(1, xlist.shape[-1] - 1):
        a = (
            mlist[..., k]
            * (xlist[..., k + 1] - xlist[..., k])
            / (ylist[..., k + 1] - ylist[..., k])
        )
        b = (
            mlist[..., k + 1]
            * (xlist[..., k + 1] - xlist[..., k])
            / (ylist[..., k + 1] - ylist[..., k])
        )
        #     mlist[...,k]*=(1.*((a**2+b**2)<9)+3.*((a**2+b**2)>=9)/torch.sqrt(a**2+b**2))
        #     mlist[...,k+1]*=(1.*((a**2+b**2)<9)+3.*((a**2+b**2)>=9)/torch.sqrt(a**2+b**2))
        clamp = (a > 3) + (b > 3)
        mlist[..., k] = (
            mlist[..., k]
            + (
                3
                * (ylist[..., k + 1] - ylist[..., k])
                / (xlist[..., k + 1] - xlist[..., k])
                - mlist[..., k]
            )
            * clamp
        )

    answer += 0.0 * (X < xlist[..., 0])
    answer += 1.0 * (X >= xlist[..., -1])
    for k in range(xlist.shape[-1] - 1):
        delta = xlist[..., k + 1] - xlist[..., k]
        t = (X - xlist[..., k]) / delta
        answer += (
            (
                ylist[..., k] * h00(t)
                + delta * mlist[..., k] * h10(t)
                + ylist[..., k + 1] * h01(t)
                + delta * mlist[..., k + 1] * h11(t)
            )
            * (X < xlist[..., k + 1])
            * (X >= xlist[..., k])
        )

    return ensure_0_1(answer)


def monotone_cubic_interpolation2(X, theta):
    while len(theta.shape) <= len(X.shape):
        theta = theta.unsqueeze(dim=-2)
    theta = torch.sigmoid(theta)

    x1 = theta[..., 0]
    x0 = -2 * torch.ones_like(x1)
    x3 = x1 + theta[..., 1]
    x2 = theta[..., 2] * (x3 - x1) + x1
    x4 = 3 * torch.ones_like(x1)

    alpha1 = 1 / 2 * (x1 - x0) * theta[..., 3]
    alpha2 = 1 / 2 * (x2 - x1) * theta[..., 4]
    alpha3 = 1 / 2 * (x3 - x2) * theta[..., 5]

    beta1 = 1 / 2 * (x2 - x1) * theta[..., 6]
    beta2 = 1 / 2 * (x3 - x2) * theta[..., 7]
    beta3 = 1 / 2 * (x4 - x3) * theta[..., 8]

    y2 = theta[..., 9]

    xlist = torch.cat(
        [
            (x1 - alpha1).unsqueeze(dim=-1),
            (x1 + beta1).unsqueeze(dim=-1),
            (x2 - alpha2).unsqueeze(dim=-1),
            (x2 + beta2).unsqueeze(dim=-1),
            (x3 - alpha3).unsqueeze(dim=-1),
            (x3 + beta3).unsqueeze(dim=-1),
        ],
        dim=-1,
    )
    zeros = torch.zeros_like(x0.unsqueeze(dim=-1))
    ylist = torch.cat(
        [
            zeros,
            (y2 / (x2 - x1) * beta1).unsqueeze(dim=-1),
            (y2 - y2 / (x2 - x1) * alpha2).unsqueeze(dim=-1),
            (y2 + (1 - y2) / (x3 - x2) * beta2).unsqueeze(dim=-1),
            (1 - (1 - y2) / (x3 - x2) * alpha3).unsqueeze(dim=-1),
            1.0 + zeros,
        ],
        dim=-1,
    )

    def monotonic_cubic_spline(x, x_data, y_data):
        dx = x_data[..., 1:] - x_data[..., :-1]
        dy = y_data[..., 1:] - y_data[..., :-1]  # Compute the slopes of the secants
        m = dy / dx  # Compute the coefficients of the monotonic cubic Hermite spline
        d0 = torch.zeros_like(m)
        d0[..., 1:] = (m[..., 1:] + m[..., :-1]) / 2
        d0[..., 0] = m[..., 0]
        d1 = m  # Adjust the slopes at knots to ensure monotonicity
        alpha = d0[..., :-1] / m[..., :-1]
        beta = d1[..., 1:] / m[..., :-1]
        tau = alpha**2 + beta**2
        d0[..., 1:-1] = ((3 / torch.sqrt(1 + tau)) * alpha * m[..., :-1])[..., 1:]
        d1[..., 1:-1] = ((3 / torch.sqrt(1 + tau)) * beta * m[..., :-1])[
            ..., 1:
        ]  # Coefficients of the cubic polynomial
        c2 = (3 * dy - 2 * d0[..., 1:] - d1[..., :-1]) / dx**2
        c3 = (
            -2 * dy + d0[..., 1:] + d1[..., :-1]
        ) / dx**3  # Compute the interpolated y values
        t = x.unsqueeze(-1) - x_data[..., :-1]
        t = torch.clamp(t, 0, float("inf"))
        y = (d1[..., :-1] + t * (2 * d0[..., 1:] - c2 * t + c3 * t**2)) * t + y_data[
            ..., :-1
        ]  # Sum the contributions of each segment
        return torch.sum(y, dim=-1)

    return monotonic_cubic_spline(X, xlist, ylist)



def lin_interpolation(X,theta):
    n_knots=theta.shape[-1]//2
    x=torch.cumsum(2/n_knots*torch.abs(theta[...,:n_knots]),dim =-1)
    y=torch.cumsum(2/n_knots*torch.abs(theta[...,n_knots:]),dim =-1)
    idx = torch.searchsorted(x, X,right=True)-1
    idx = torch.where(idx >= x.shape[-1]-1, x.shape[-1]-2, idx)
    return torch.clamp(y[...,idx]+(y[...,idx+1]-y[...,idx])/(x[...,idx+1]-x[...,idx])*(X - x[...,idx]),0.,1.)




class DirectCDFDistrib(torch.distributions.Distribution):
    def __init__(
        self,
        func,
        theta,
        device,
        samples_cdf_tolerance=1e-3,
        max_samples_iter=1000,
        lower_samples_bound=-10,
        upper_samples_bound=10,
    ):
        super().__init__()
        self.func = func
        self.theta = theta
        self.arg_constraints = {}
        self.device = device

        self.samples_cdf_tolerance = samples_cdf_tolerance
        self.max_samples_iter = max_samples_iter
        self.lower_samples_bound = lower_samples_bound
        self.upper_samples_bound = upper_samples_bound

    def cdf(self, x):
        return self.func(x, self.theta)

    def log_prob(self, x):
        y = self.cdf(x)
        y.backward()
        return torch.log(x.grad)

    def sample(self, sample_shape, verbose=True):
        p = torch.rand(*(self.theta.shape[:-1]), *sample_shape, 1).to(self.device)
        X = (
            (self.upper_samples_bound + self.lower_samples_bound)
            / 2
            * torch.ones_like(p)
        )
        left = self.lower_samples_bound * torch.ones_like(p)
        right = self.upper_samples_bound * torch.ones_like(p)
        i = 0
        crit = torch.Tensor([1e9])
        while (
            i < self.max_samples_iter
            and (torch.abs(crit) > self.samples_cdf_tolerance).any()
        ):
            i += 1
            crit = self.cdf(X) - p

            go_right = crit <= 0
            go_left = ~go_right

            U = X
            X = go_right * right / 2 + go_left * left / 2 + U / 2
            right = go_right * right + go_left * U
            left = go_left * left + go_right * U

        if verbose:
            print(
                f"Sampling with shape {sample_shape} in {i} iterations with a mean crit of {torch.abs(crit).mean():.4f}, a max crit of {torch.abs(crit).max():.4f}, proportion of samples below the tolerance {(100.*(torch.abs(crit)<self.samples_cdf_tolerance)).mean():.2f}%"
            )

        return X


class DirectCDFfromNN(torch.nn.Module):
    def __init__(self, neural_net, func, device):
        super().__init__()
        self.neural_net = neural_net.to(device)
        self.func = func
        self.device = device

    def forward(self, features):
        return DirectCDFDistrib(
            self.func, self.neural_net(features), device=self.device
        )
