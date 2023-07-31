"""
Constructs a network from layers and can load weights to resume network training
"""
import json
import logging as log

import torch
from torch import nn, optim, Tensor

from msapnet.autoencoder import layers


class Network(nn.Module):
    """
    Constructs a neural network from a configuration file

    Attributes
    ----------
    encoder : boolean
        If network is an encoder
    name : string
        Name of the network, used for saving
    layers : list[dictionary]
        Layers with layer parameters
    network : ModuleList
        Network construction
    optimizer : Optimizer
        Network optimizer
    scheduler : ReduceLROnPlateau
        Optimizer scheduler

    Methods
    -------
    forward(x)
        Forward pass of CNN
    """

    def __init__(
        self,
        spectra_size: int,
        params_size: int,
        learning_rate: float,
        name: str,
        config_dir: str,
        aux_size: int = 0,
        degenearcy: int = 0,
    ):
        """
        Parameters
        ----------
        spectra_size : integer
            Size of the input tensor
        params_size : integer
            Size of the output tensor
        learning_rate : float
            Optimizer initial learning rate
        name : string
            Name of the network, used for saving
        config_dir : string
            Path to the network config directory
        """
        super().__init__()
        self.name = name

        # If network is an encoder
        if "Encoder" in name:
            self.encoder = True
            input_size = spectra_size + aux_size
            output_size = params_size + degenearcy
        else:
            self.encoder = False
            input_size = params_size + aux_size
            output_size = spectra_size

        # Construct layers in CNN
        self.layers, self.network, self.loss_weights = create_network(
            input_size,
            output_size,
            f"{config_dir}{name}.json",
        )

        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            verbose=True,
        )
        self.learning_rate = learning_rate

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the CNN

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Output tensor from the CNN
        """
        outputs = [x]

        for i, layer in enumerate(self.layers):
            # Concatenation layers
            if layer["type"] == "concatenate":
                x = torch.cat((outputs[layer["layer"]][:, layer["indexes"]], x), dim=1)
            # Shortcut layers
            elif layer["type"] == "shortcut":
                x = x + outputs[layer["layer"]]
            # use_indexes layers
            elif layer["type"] == "use_indexes":
                x = x[:, layer["indexes"]]
            elif layer["type"] == "truncate":
                x = x[:, layer["index"] :]
            elif layer["type"] == "create_side_network":
                x = outputs[layer["layer"]][:, layer["indexes"]]
            elif layer["type"] == "end_side_network":
                x = outputs[layer["layer"]]  # removed the suspicious +1
            elif layer["type"] == "gru":
                try:
                    if layer[
                        "inverting"
                    ]:  # inverting the second and third dimensions to have the recurrency on the spectrum not the filters
                        x = torch.transpose(x, 1, 2)
                        x = self.network[i](x)
                        x = torch.transpose(x, 1, 2)
                    else:
                        x = self.network[i](x)
                except KeyError:
                    x = torch.transpose(x, 1, 2)
                    x = self.network[i](x)
                    x = torch.transpose(x, 1, 2)
            elif layer["type"] == "duplicate":
                x = x.unsqueeze(dim=1).repeat(1, layer["length"], 1)
            # All other layers
            else:
                x = self.network[i](x)

            # print(i)
            # print(layer['type'])
            # print(x.shape)
            # try:
            #     print(outputs[layer['layer']].shape)
            # except KeyError:
            #     print('')

            outputs.append(x)
        # raise KeyboardInterrupt
        return x


class CombinedDecoder(nn.Module):
    def __init__(self, decoder_scale: nn.Module, decoder_shape: nn.Module):
        super().__init__()
        self.name = (
            "CombinedDecoder : " + decoder_scale.name + " | " + decoder_shape.name
        )
        self.decoder_scale = decoder_scale
        self.decoder_shape = decoder_shape
        self.truncate_index = 1
        self.scale_loss_weight = 1e-2

        self.optimizer = optim.Adam(
            self.parameters(), lr=decoder_shape.learning_rate, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            verbose=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        _scale = self.decoder_scale(x)
        _shape = self.decoder_shape(x)
        return torch.cat((_scale, _shape), dim=1)


class MSDecoder(nn.Module):
    def __init__(
        self,
        combineddecoder: CombinedDecoder,
        num_scenarios: int = 5,
        is_degeneracy_predicted: bool = True,
    ):
        super().__init__()
        self.name = f"MSDecoder {num_scenarios} sc : {combineddecoder.name}"
        self.decoder = combineddecoder
        self.num_scenarios = num_scenarios
        self.is_degeneracy_predicted = is_degeneracy_predicted

    def forward(self, x):
        y = []
        if self.is_degeneracy_predicted:
            for i in range(x.shape[1]):
                y.append(
                    self.decoder.forward(x[:, i, :-1]).unsqueeze(dim=1)
                )  # if we predict a degeneracy, we must remove the last predicted parameter
            return torch.cat(y, dim=1)

        for i in range(x.shape[1]):
            y.append(self.decoder.forward(x[:, i, :]).unsqueeze(dim=1))
        return torch.cat(y, dim=1)


def create_network(
    input_size: int, output_size: int, config_path: str
) -> tuple[list[dict], nn.ModuleList]:
    """
    Creates a network from a config file

    Parameters
    ----------
    input_size : integer
        Size of the input
    output_size : integer
        Size of the spectra
    config_path : string
        Path to the config file

    Returns
    -------
    tuple[list[dictionary], ModuleList]
        Layers in the network with parameters and network construction
    """
    # Load network configuration file
    with open(config_path, "r", encoding="utf-8") as file:
        file = json.load(file)

    # Initialize variables
    kwargs = {
        "data_size": input_size,
        "output_size": output_size,
        "dims": [input_size],
        "dropout_prob": file["net"]["dropout_prob"],
    }
    module_list = nn.ModuleList()

    # Create layers
    for i, layer in enumerate(file["layers"]):
        kwargs["i"] = i
        kwargs["module"] = nn.Sequential()

        try:
            kwargs = getattr(layers, layer["type"])(kwargs, layer)
        except AttributeError as error:
            log.error(f"Unknown layer: {layer['type']}")
            raise error

        module_list.append(kwargs["module"])

    # Create loss weights dictionnary

    loss_weights = {}
    try:
        loss_weights["latent_loss_weight"] = file["net"]["latent_loss_weight"]
    except KeyError:
        loss_weights["latent_loss_weight"] = 1e-3
        # print(f"No latent loss weight specified in config file, defaulting to {loss_weights['latent_loss_weight']}")

    try:
        loss_weights["sum_to_1_loss_weight"] = file["net"]["sum_to_1_loss_weight"]
    except KeyError:
        loss_weights["sum_to_1_loss_weight"] = 1e-4
        # print(f"No sum to 1 loss weight specified in config file, defaulting to {loss_weights['sum_to_1_loss_weight']}")

    try:
        loss_weights["shape_loss_weight"] = file["net"]["shape_loss_weight"]
    except KeyError:
        loss_weights["shape_loss_weight"] = 1

    try:
        loss_weights["scale_loss_weight"] = file["net"]["scale_loss_weight"]
    except KeyError:
        loss_weights["scale_loss_weight"] = 1

    try:
        loss_weights["proximity_loss_weight"] = file["net"]["proximity_loss_weight"]
    except KeyError:
        loss_weights["proximity_loss_weight"] = 1e-4

    try:
        loss_weights["cst_dgen"] = file["net"]["cst_dgen"]
    except KeyError:
        loss_weights["cst_dgen"] = 1e-2

    print(f"list of dims of the network: {kwargs['dims']}")

    return file["layers"], module_list, loss_weights


def load_network(
    load_name: str, states_dir: str, network: Network
) -> tuple[int, Network, tuple[list, list]]:
    """
    Loads the network from a previously saved state

    Can account for changes in the network

    Parameters
    ----------
    load_name : integer
        File number of the saved state
    states_dir : string
        Directory to the save files
    network : Network
        The network to append saved state to

    Returns
    -------
    tuple[int, Encoder | Decoder, Optimizer, ReduceLROnPlateau, tuple[list, list]]
        The initial epoch, the updated network, optimizer
        and scheduler, and the training and validation losses
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_state = torch.load(
        f"{states_dir}{network.name.split()[0]}_{load_name}.pth", map_location=device
    )

    # Apply the saved states to the new network
    initial_epoch = d_state["epoch"]
    network.load_state_dict(network.state_dict() | d_state["state_dict"])
    network.optimizer.load_state_dict(d_state["optimizer"])
    network.scheduler.load_state_dict(d_state["scheduler"])
    train_loss = d_state["train_loss"]
    val_loss = d_state["val_loss"]

    return initial_epoch, network, (train_loss, val_loss)
