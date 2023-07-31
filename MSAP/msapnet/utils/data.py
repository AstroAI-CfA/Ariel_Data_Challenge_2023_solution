"""
Loads data and creates data loaders for autoencoders training.
"""
import torch
import numpy as np
from numpy import ndarray
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from msapnet.utils.utils import even_length


class SpectrumDataset(Dataset):
    """
    A dataset object containing spectrum data for PyTorch training

    Attributes
    ----------
    log_params : list
        Index of each free parameter in logarithmic space
    transform : list[tuple[ndarray, ndarray]]
        Min and max spectral range and mean & standard deviation of parameters
    names : ndarray
        Names of each spectrum
    spectra : tensor
        Spectra dataset
    uncertainty : tensor
        Spectral Poisson uncertainty
    params : tensor
        Parameters for each spectra if supervised
    indices : ndarray, default = None
        Data indices for random training & validation datasets

    Methods
    -------
    downscaler(downscales)
        Downscales input spectra
    """

    def __init__(
        self,
        data_file: str,
        params_path: str,
        aux_path: str,
        noise_path: str,
        log_params: list[int],
        names_path: str = None,
        transform_ft=None,
        transform_tg=None,
        transform_au=None,
        transform_no=None,
        device=torch.device("cpu"),
    ):
        """
        Parameters
        ----------
        data_file : string
            Path to the file with the spectra dataset
        params_path : string
            Path to the labels file, if none, then an unsupervised approach is used
        log_params : list[int]
            Index of each free parameter in logarithmic space
        names_path : string, default = None
            Path to the names of the spectra, if none, index value will be used
        transform : [transfrom, inverse transform]
            function for transforming and inverse transforming the data
            _ft: for features (spectra)
            _tg: for targets (params)
        """
        self.indices = None
        self.log_params = log_params

        spectra = torch.load(data_file).to(device)
        self.spectra = transform_ft(spectra.clone())

        # If no parameters are provided
        if not params_path:
            self.params = np.empty(self.spectra.size(0))
            self.transform_ft = transform_ft
            self.transform_tg = transform_tg
            self.names = np.arange(self.spectra.size(0))
            return

        # Load spectra parameters and names
        params = torch.load(params_path).to(device)

        if names_path:
            self.names = np.load(names_path)
        else:
            self.names = np.arange(self.spectra.size(0))

        # Transform parameters
        self.params = transform_tg(params.clone())

        # Load and transform auxiliary variables
        if aux_path is not None:
            aux = torch.load(aux_path).to(device)
            
            # TO REMOVE: this was added for the sole purpose of the Ariel data challenge, to add some tricks. IT SHOULD BE REMOVED!
            R_p_estimated_spectrum = (
                aux[:, [2]] / 69911000 * torch.sqrt(torch.mean(spectra, dim=1, keepdim=True))
            )
            R_p_estimated_gravity = torch.sqrt((aux[:, [4]] / aux[:, [7]] * 6.674e-11)) / 69911000
            in_vertical_line = (
                1.0
                * (0.71488e8 / 69911000 < R_p_estimated_gravity)
                * (R_p_estimated_gravity < 0.714881e8 / 69911000)
            )
            R_p_estimated_combo = (
                R_p_estimated_gravity * (1 - in_vertical_line)
                + R_p_estimated_spectrum * in_vertical_line
            )
            aux = torch.cat(
                [
                    aux,
                    R_p_estimated_spectrum,
                    R_p_estimated_gravity,
                    in_vertical_line,
                    R_p_estimated_combo,
                ],
                dim=1,
            )  #
            
            self.aux = transform_au(aux)
        else:
            self.aux = None

        # Load and transform noises
        if noise_path is not None:
            noise = torch.load(noise_path).to(device)
            self.noise = transform_no(noise)
        else:
            self.noise = None

        self.transform_ft = transform_ft
        self.transform_tg = transform_tg
        self.transform_au = transform_au
        self.transform_no = transform_no

    def __len__(self) -> int:
        return self.spectra.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, int]:
        """
        Gets the training data for a given index

        Parameters
        ----------
        idx : integer
            Index of the target spectrum

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, string | integer]
            Spectrum data, target parameters, spectrum uncertainty and spectrum name/number
        """
        if self.aux is not None:
            aux = self.aux[idx]
        else:
            aux = torch.zeros(0)
        if self.noise is not None:
            noise = self.noise[idx]
        else:
            noise = torch.zeros(0)

        return self.spectra[idx], self.params[idx], aux, noise, self.names[idx]

    def _min_clamp(self, data: ndarray) -> ndarray:
        """
        Clamps all values <= 0 to the minimum non-zero positive value

        Parameters
        ----------
        data : ndarray
            Input data to be clamped

        Returns
        -------
        ndarray
            Clamped data
        """
        data = np.swapaxes(data, 0, 1)
        min_count = np.min(
            data,
            where=data > 0,
            initial=np.max(data),
            axis=0,
        )

        return np.swapaxes(np.maximum(data, min_count), 0, 1)

    def downscaler(self, downscales: int):
        """
        Downscales input spectra

        Parameters
        ----------
        downscales : integer
            Number of times to downscale
        """
        avgpool = nn.AvgPool1d(kernel_size=2)

        self.spectra = self.spectra.unsqueeze(dim=1)

        for _ in range(downscales):
            self.spectra = avgpool(self.spectra)

        self.spectra = self.spectra.squeeze(dim=1)


def load_x_data(y_size: int, config: dict):
    """
    Fetches x data from file and matches the length to the y data

    Parameters
    ----------
    y_size : int
        Number of y data points

    Returns
    -------
    ndarray
        x data points
    """
    x_data = (
        torch.tensor(
            torch.load(f"{config['path']['local-path']}{config['path']['x-data']}")
        )
        .detach()
        .cpu()
        .numpy()
    )

    # Make sure x data size is even
    if x_data.size % 2 != 0:
        x_data = np.append(x_data[:-2], np.mean(x_data[-2:]))

    # Make sure x data size is of the same size as y data
    if x_data.size != y_size and x_data.size % y_size == 0:
        x_data = x_data.reshape(int(x_data.size / y_size), -1)
        x_data = np.mean(x_data, axis=0)

    return x_data


def data_initialisation(
    spectra_path: str,
    params_path: str,
    aux_path: str,
    noise_path: str,
    log_params: list,
    kwargs: dict,
    val_frac: float = 0.1,
    names_path: str = None,
    transform_ft=None,
    transform_tg=None,
    transform_au=None,
    transform_no=None,
    indices: ndarray = None,
    device=torch.device("cpu"),
) -> tuple[DataLoader, DataLoader]:
    """
    Initialises training and validation data

    Parameters
    ----------
    spectra_path : string
        Path to synthetic data
    params_path : string
        Path to labels
    log_params : list
        Index of each free parameter in logarithmic space
    kwargs : dictionary
        Keyword arguments for dataloader
    val_frac : float, default = 0.1
        Fraction of validation data
    names_path : string, default = None
        Path to the names of the spectra, if none, index value will be used
    transform : [transfrom, inverse transform]
            function for transforming and inverse transforming the data
            _ft: for features (spectra)
            _tg: for targets (params)
    indices : ndarray, default = None
        Data indices for random training & validation datasets

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Dataloaders for the training and validation datasets
    """
    batch_size = 120

    # Fetch dataset & create train & val data
    dataset = SpectrumDataset(
        data_file=spectra_path,
        params_path=params_path,
        aux_path=aux_path,
        noise_path=noise_path,
        log_params=log_params,
        names_path=names_path,
        transform_ft=transform_ft,
        transform_tg=transform_tg,
        transform_au=transform_au,
        transform_no=transform_no,
        device=device,
    )
    val_amount = max(int(len(dataset) * val_frac), 1)

    # If network hasn't trained on data yet, randomly separate training and validation
    if indices is None or indices.size != len(dataset):
        indices = np.random.choice(len(dataset), len(dataset), replace=False)

    dataset.indices = indices

    train_dataset = Subset(dataset, indices[:-val_amount])
    val_dataset = Subset(dataset, indices[-val_amount:])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    if val_frac == 0:
        val_loader = train_loader
    else:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )

    print(
        f"Training data size: {len(train_dataset)}\tValidation data size: {len(val_dataset)}"
    )

    return train_loader, val_loader
