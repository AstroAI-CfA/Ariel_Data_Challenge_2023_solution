"""
Defines transforms for feature engineering.
Beware that some transforms have to store some tensors so they can be RAM consuming. And also they should be changed when changing the dataset tensors.
"""

import torch
import numpy as np


class Transform:
    def __init__(self):
        self.is_defined = False
        self.is_inverse_defined = False

    def __call__(self, data):
        raise NotImplementedError

    def inverse(self, data):
        raise NotImplementedError


class ScalarTransform(Transform):
    def __init__(self, _min, _scale):
        super().__init__()
        self.min = _min
        self.scale = _scale
        self.is_defined = True
        self.is_inverse_defined = True

    def __call__(self, data):
        _min = self.min
        _scale = self.scale

        if (
            isinstance(_min, torch.Tensor) and len(_min.shape) == 1
        ):  # if min is a tensor of only one dimension, we have to expand its dims to reach that of the data
            if (
                _min.shape[0] == data.shape[0]
            ):  # if its the batch dim that is equal to that of the data, we expand the dims on the right side (such as features dims, samples dims, etc...)
                for j in range(len(_min.shape), len(data.shape)):
                    _min = _min.unsqueeze(dim=-1)
            elif (
                _min.shape[-1] == data.shape[-1]
            ):  # if tis the feature dim that is equal to that of the data, we expand the dims on the right side (such as batch dim, samples dims, etc...)
                for j in range(len(_min.shape), len(data.shape)):
                    _min = _min.unsqueeze(dim=0)

        if (
            isinstance(_scale, torch.Tensor) and len(_scale.shape) == 1
        ):  # we do the same thing for the scale
            if _scale.shape[0] == data.shape[0]:
                for j in range(len(_scale.shape), len(data.shape)):
                    _scale = _scale.unsqueeze(dim=-1)
            elif _scale.shape[-1] == data.shape[-1]:
                for j in range(len(_scale.shape), len(data.shape)):
                    _scale = _scale.unsqueeze(dim=0)

        return (data - _min) / _scale

    def inverse(self, data):
        _min = self.min
        _scale = self.scale

        if (
            isinstance(_min, torch.Tensor) and 0 < len(_min.shape) < len(data.shape)
        ):  # if min is a tensor of only one dimension, we have to expand its dims to reach that of the data
            if (
                _min.shape[0] == data.shape[0]
            ):  # if its the batch dim that is equal to that of the data, we expand the dims on the right side (such as features dims, samples dims, etc...)
                for j in range(len(_min.shape), len(data.shape)):
                    _min = _min.unsqueeze(dim=-1)
            elif (
                _min.shape[-1] == data.shape[-1]
            ):  # if tis the feature dim that is equal to that of the data, we expand the dims on the right side (such as batch dim, samples dims, etc...)
                for j in range(len(_min.shape), len(data.shape)):
                    _min = _min.unsqueeze(dim=0)

        if (
            isinstance(_scale, torch.Tensor) and 0 < len(_scale.shape) < len(data.shape)
        ):  # we do the same thing for the scale
            if _scale.shape[0] == data.shape[0]:
                for j in range(len(_scale.shape), len(data.shape)):
                    _scale = _scale.unsqueeze(dim=-1)
            elif _scale.shape[-1] == data.shape[-1]:
                for j in range(len(_scale.shape), len(data.shape)):
                    _scale = _scale.unsqueeze(dim=0)
        
        return data * _scale + _min


class ComposedTransforms(Transform):
    def __init__(self, transforms_list):
        super().__init__()
        self.transforms_list = transforms_list
        self.is_defined = True
        self.is_inverse_defined = True
        for t in transforms_list:
            if not t.is_defined:
                self.is_defined = False
            if not t.is_inverse_defined:
                self.is_defined = False

    def __call__(self, data):
        for t in self.transforms_list:
            data = t(data)
        return data

    def inverse(self, data):
        for i in range(len(self.transforms_list) - 1, -1, -1):
            data = self.transforms_list[i].inverse(data)
        return data


class IndexTransform(Transform):
    def __init__(self, indexes, transform):
        super().__init__()
        self.indexes = indexes
        self.transform = transform
        self.is_defined = True
        self.is_inverse_defined = True

    def __call__(self, data):
        X = data.clone()
        X[..., self.indexes] = self.transform(X[..., self.indexes])
        return X

    def inverse(self, data):
        X = data.clone()
        X[..., self.indexes] = self.transform.inverse(X[..., self.indexes])
        return X


class LogTransform(Transform):
    def __init__(self):
        super().__init__()
        self.is_defined = True
        self.is_inverse_defined = True

    def __call__(self, data):
        return torch.log10(data)

    def inverse(self, data):
        return 10 ** (data)


class LogTransform_uncertainty(Transform):
    def __init__(self, data):
        super().__init__()
        self.is_defined = True
        self.is_inverse_defined = True
        self.data = data

    def __call__(self, noise):
        a = noise / self.data  # we have to check that the log is well defined
        a1 = (a > -1) * a
        a2 = (a < 1) * a
        return 1 / ((a > -1) + (a < 1)) * (torch.log10(1 + a1) - torch.log10(1 - a2))

    def inverse(self, noise):
        return 10 ** (self.data) / 2 * (10**noise - 10 ** (-noise))


class ExpTransform(Transform):
    def __init__(self, alpha=1):
        self.alpha = alpha
        super().__init__()
        self.is_defined = True
        self.is_inverse_defined = True

    def __call__(self, data):
        return torch.exp(data * self.alpha)

    def inverse(self, data):
        return torch.log(data) / self.alpha


class Scale_each(Transform):
    def __init__(self, index=None):
        # set index to None to get the scale by any index
        super().__init__()
        self.is_defined = True
        self.is_inverse_defined = True
        self.index = index

    def __call__(self, data):
        if self.index is not None:
            _scale = torch.abs(data[..., self.index])
        else:
            _scale, _ = (torch.abs(data)).max(dim=-1)
        _scale = _scale.unsqueeze(dim=-1)
        data /= _scale
        return torch.cat([_scale, data], dim=1)

    def inverse(self, data):
        _scale = data[..., 0]
        return data[..., 1:] * _scale


class Normalise_each(Transform):
    """
    Normalise each spectrum and concatenate the normalising features before the spectrum.
    We can provide the normalising scheme in ft:
        'min_max': store min and max, subtract min and divide by (max-min)
        'min_scale': store min and scale=(max-min), subtract min and divide by scale
        'mean_std': store mean and std, subtract mean and divide by std
    """

    def __init__(self, ft: str = "min_max"):
        super().__init__()
        self.is_defined = True
        self.is_inverse_defined = True
        self.ft = ft  # the normalising feature

    def __call__(self, data):
        if (
            "min" in self.ft.lower()
        ):  # if min is in the normalising scheme, we get the min and max of each spectrum
            _min, _ = data.min(dim=-1, keepdim=True)
            _max, _ = data.max(dim=-1, keepdim=True)
            data = (data - _min) / (
                _max - _min
            )  # we normalise the data using the min and max
            if (
                "max" in self.ft.lower()
            ):  # if max is in the normalising scheme, we store min and max
                return torch.cat([_min, _max, data], dim=-1)
            if (
                "scale" in self.ft.lower()
            ):  # if scale is in the normalising scheme, we store min and scale
                return torch.cat([_min, _max - _min, data], dim=-1)
        if (
            "mean" in self.ft.lower()
        ):  # otherwise if mean is in the normalising scheme, we get the mean
            _mean = data.mean(dim=-1, keepdim=True)
            if "std" in self.ft.lower():  # we get the std
                _std = data.std(dim=1, keepdim=True)
                data = (data - _mean) / _std  # we normalise using mean and std
                return torch.cat([_mean, _std, data], dim=-1)  # we store them
        raise KeyError(
            self.ft
        )  # we raise error if the normalising scheme is not recognised

    def inverse(self, data):
        if "min" in self.ft.lower():
            _min = data[..., 0]
            if "max" in self.ft.lower():
                _max = data[..., 1]
                return (_max - _min) * data[..., 2:] + _min
            if "scale" in self.ft.lower():
                _scale = data[..., 1]
                return _scale * data[..., 2:] + _min
        if "mean" in self.ft.lower():
            _mean = data[..., 0]
            if "std" in self.ft.lower():
                _std = data[..., 1]
                return _std * data[..., 2:] + _mean
        raise KeyError(self.ft)


class Normalise_each_uncertainty(Transform):
    """
    In the case where a Normalise_each is being used on the spectra, the uncertainty has to be transformed consequently.
    This Transform transforms the uncertainty accordingly using the data to define the scale coefficients.
    """

    def __init__(self, data, ft: str = "min_max"):
        super().__init__()
        self.is_defined = True
        self.is_inverse_defined = True
        self.ft = ft
        if "min" in self.ft.lower():
            self._min, _ = data.min(dim=-1, keepdim=True)
            self._max, _ = data.max(dim=-1, keepdim=True)
        elif "mean" in self.ft.lower():
            # self._mean=data.mean(dim=1,keepdim=True) # we don't need the mean for transforming the noise
            if "std" in self.ft.lower():
                self._std = data.std(dim=-1, keepdim=True)
            else:
                raise KeyError(self.ft)
        else:
            raise KeyError(self.ft)

    def __call__(self, noise):
        if "min" in self.ft.lower():
            _noise_min = noise.mean(
                dim=-1, keepdim=True
            )  # TODO: define the _noise_min, this is NOT CORRECT!
            if "max" in self.ft.lower():
                _noise_max = noise.mean(
                    dim=-1, keepdim=True
                )  # TODO: define the _noise_max, this is NOT CORRECT!
                return torch.cat(
                    [_noise_min, _noise_max, noise / (self._max - self._min)], dim=-1
                )
            if "scale" in self.ft:
                _noise_scale = torch.sqrt(
                    _noise_min**2 + _noise_max**2
                )  # sum of variance of independent variables
                return torch.cat(
                    [_noise_min, _noise_scale, noise / (self._max - self._min)], dim=-1
                )
        if "mean" in self.ft.lower():
            _noise_mean = torch.sqrt(
                (noise**2).mean(dim=-1, keepdim=True)
            )  # mean of variance of independent variables
            if "std" in self.ft.lower():
                _noise_std = (
                    torch.zeros_like(_noise_mean) + 1e-3
                )  # TODO: define the _noise_std, this is NOT CORRECT!
                return torch.cat([_noise_mean, _noise_std, noise / self._std], dim=-1)
        raise KeyError(self.ft)

    def inverse(self, noise):
        if "min" in self.ft.lower():
            return (self._max - self._min) * noise[..., 2:]
        if "mean" in self.ft.lower():
            if "std" in self.ft.lower():
                return self._std * noise[..., 2:]
        raise KeyError(self.ft)


class Mean_each(Transform):
    def __init__(self):
        super().__init__()
        self.is_defined = True
        self.is_inverse_defined = True

    def __call__(self, data):
        _mean = torch.mean(data, dim=-1, keepdim=True)
        return torch.cat([_mean, data - _mean], dim=-1)

    def inverse(self, data):
        _mean = data[..., 0]
        return data[..., 1:] + _mean


class Mean_each_uncertainty(Transform):
    def __init__(self):
        super().__init__()
        self.is_defined = True
        self.is_inverse_defined = True

    def __call__(self, noise):
        _noise_mean = torch.sqrt(torch.mean(noise**2, dim=-1, keepdim=True))
        return torch.cat([_noise_mean, noise], dim=-1)

    def inverse(self, noise):
        return noise[..., 1:]


class Mean_zero(Transform):
    def __init__(self):
        super().__init__()
        self.is_defined = True
        self.is_inverse_defined = False

    def __call__(self, data):
        if len(data.shape) == 1:
            return data - data.mean()
        return data - data.mean(dim=-1, keepdim=True)

    def inverse(self, data):
        print("Mean zero currently not invertible")
        raise NotImplementedError


class Truncate(Transform):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.is_defined = True
        self.is_inverse_defined = False

    def __call__(self, data):
        return data[..., self.index :]

    def inverse(self, data):
        print("Truncate is not invertible")
        raise NotImplementedError


class UseIndex(Transform):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.is_defined = True
        self.is_inverse_defined = False

    def __call__(self, data):
        if isinstance(self.index, int):
            return data[:, self.index : self.index + 1]
        return data[:, self.index]

    def inverse(self, data):
        print("UseIndex is not invertible")
        raise NotImplementedError


class InverseTransform(Transform):
    def __init__(self, transform):
        super().__init__()
        self.inverse_transform = transform
        self.is_inverse_defined = self.inverse_transform.is_defined
        self.is_defined = self.inverse_transform.is_inverse_defined

    def __call__(self, data):
        return self.inverse_transform.inverse(data)

    def inverse(self, data):
        return self.inverse_transform(data)


class SigmoidTransform(Transform):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.is_defined = True
        self.is_inverse_defined = True

    def __call__(self, data):
        return torch.sigmoid(self.alpha * data)

    def inverse(self, data):
        return torch.logit(data) / self.alpha


def normalisation_uniform_transform(data):
    """Create a transform and inverse transform for normalisation between 0 and 1

    Args:
        data (tensor, ndarray): data to use to define the normalisation

    Returns:
        ScalarTransform (Transform): transform for the normalisation
    """
    _min = data.min()
    _scale = data.max() - _min

    return ScalarTransform(_min, _scale)


def normalisation_specific_transform(data):
    l = len(data.shape)
    _min, _ = data.min(dim=0)
    _max, _ = data.max(dim=0)
    for j in range(1, l - 1):  # reduce each dimension except the last one
        _min, _ = _min.min(dim=0)
        _max, _ = _max.max(dim=0)
    _scale = _max - _min

    return ScalarTransform(_min, _scale)
