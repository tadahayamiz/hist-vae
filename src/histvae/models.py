# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019
 
Histogram Variational Autoencoder (VAE) for 1D, 2D, and 3D data.

route 6
- change the decoder last layer to relu
- change recon_loss to MSE

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import inspect

from .optimal_transport import JointSinkhornDivergence, validate_ot_config


DECODER_OUTPUT_MODES = ("legacy_sigmoid", "simplex_softmax")
RECONSTRUCTION_LOSSES = ("mse", "forward_kl")
CONDITION_MODES = ("none", "decoder")


def validate_decoder_output_mode(decoder_output_mode):
    if decoder_output_mode not in DECODER_OUTPUT_MODES:
        raise ValueError(
            f"Unsupported decoder_output_mode: {decoder_output_mode!r}. "
            "Use 'legacy_sigmoid' or 'simplex_softmax'."
        )
    return decoder_output_mode


def validate_reconstruction_loss(reconstruction_loss):
    if reconstruction_loss not in RECONSTRUCTION_LOSSES:
        raise ValueError(
            f"Unsupported reconstruction_loss: {reconstruction_loss!r}. "
            "Use 'mse' or 'forward_kl'."
        )
    return reconstruction_loss


def validate_condition_mode(condition_mode):
    if condition_mode not in CONDITION_MODES:
        raise ValueError(
            f"Unsupported condition_mode: {condition_mode!r}. "
            "Use 'none' or 'decoder'."
        )
    return condition_mode

# Enum for clearly managing data dimensions
class DataDim(Enum):
    ONE_D = 1
    TWO_D = 2
    THREE_D = 3

# functions to get the number of dimensions from the input shape
def get_conv_layer(dim: DataDim):
    return {DataDim.ONE_D: nn.Conv1d, DataDim.TWO_D: nn.Conv2d, DataDim.THREE_D: nn.Conv3d}[dim]

def get_conv_transpose_layer(dim: DataDim):
    return {DataDim.ONE_D: nn.ConvTranspose1d, DataDim.TWO_D: nn.ConvTranspose2d, DataDim.THREE_D: nn.ConvTranspose3d}[dim]

def get_batchnorm_layer(dim: DataDim):
    return {DataDim.ONE_D: nn.BatchNorm1d, DataDim.TWO_D: nn.BatchNorm2d, DataDim.THREE_D: nn.BatchNorm3d}[dim]


# Encoder block (convolution + BatchNorm + activation + residual connection)
class ResidualConvBlock(nn.Module):
    """
    Convolutional block with residual connection (for Encoder)

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dim, dropout_conv=0.3):
        super().__init__()
        Conv = get_conv_layer(dim)
        BatchNorm = get_batchnorm_layer(dim)
        self.conv1 = Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout_conv) if dropout_conv > 0 else nn.Identity()
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm(out_channels)
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            # dimension adjustment for skip connection (if needed)
            self.skip = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, stride=stride),
                BatchNorm(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Apply dropout after the first convolution
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResidualConvTransposeBlock(nn.Module):
    """
    Deconvolutional block with residual connection (for Decoder)
 
    """
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding,
            dim, dropout_conv=0.3, activation="relu"
            ):
        super().__init__()
        ConvT = get_conv_transpose_layer(dim)
        BatchNorm = get_batchnorm_layer(dim)
        self.convt1 = ConvT(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout_conv) if dropout_conv > 0 else nn.Identity()
        self.convt2 = ConvT(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm(out_channels)
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                ConvT(in_channels, out_channels, kernel_size=stride, stride=stride),
                BatchNorm(out_channels)
            )
        # ReLU by default, but can be changed to Sigmoid for the last layer
        act_layer = nn.ReLU(inplace=True) if activation == "relu" else nn.Sigmoid()
        self.activation = act_layer

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.convt1(x)))
        out = self.dropout(out)  # Apply dropout after the first deconvolution
        out = self.bn2(self.convt2(out))
        out += identity
        return self.activation(out)


# VAE Encoder to latent space (encoding)
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, dim, dropout_conv=0.3):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(
                ResidualConvBlock(
                    in_channels, h_dim, kernel_size=3, stride=2, padding=1,
                    dim=dim, dropout_conv=dropout_conv
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

# VAE Decoder to reconstruct the input (decoding)
class Decoder(nn.Module):
    def __init__(
            self, out_channels, hidden_dims, dim, dropout_conv=0.3,
            output_mode="legacy_sigmoid"
            ):
        super().__init__()
        self.output_mode = validate_decoder_output_mode(output_mode)
        hidden_dims = hidden_dims[::-1]
        layers = []
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(
                ResidualConvTransposeBlock(
                    in_channels, h_dim, kernel_size=4, stride=2, padding=1,
                    dim=dim, dropout_conv=dropout_conv
                )
            )
            in_channels = h_dim
        if self.output_mode == "legacy_sigmoid":
            layers.append(
                ResidualConvTransposeBlock(
                    in_channels, out_channels, kernel_size=4, stride=2,
                    padding=1, activation="sigmoid", dim=dim,
                    dropout_conv=dropout_conv
                    )
                # histogram data is normalized to [0, 1]
                )
            self.decoder = nn.Sequential(*layers)
            self.output_layer = None
        else:
            ConvT = get_conv_transpose_layer(dim)
            self.decoder = nn.Sequential(*layers)
            self.output_layer = ConvT(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )

    def forward(self, x):
        x = self.decoder(x)
        if self.output_mode == "legacy_sigmoid":
            return x
        logits = self.output_layer(x)
        probability = torch.softmax(logits.flatten(start_dim=1), dim=1)
        return probability.view_as(logits)

# main ConvVAE class
class ConvVAE(nn.Module):
    def __init__(
            self, input_shape=None, latent_dim=128, hidden_dims=None,
            dropout_conv=0.3, decoder_output_mode="legacy_sigmoid",
            reconstruction_loss="mse", ot_loss="none", ot_weight=0.0,
            ot_p=1, ot_blur=0.05, ot_scaling=0.8,
            ot_backend="tensorized", ot_mass_epsilon=0.0,
            ot_support=None, condition_mode="none", condition_dim=0
            ):
        """
        Variational Autoencoder (VAE) for 1D, 2D, and 3D data.

        Parameters
        ----------
        input_shape: tuple
            Shape of the input data like (channels, height, width) or (channels, length)

        latent_dim: int
            Dimension of the latent space

        hidden_dims: list of int
            List of hidden dimensions for the encoder and decoder

        dropout_conv: float
            Dropout probability inside convolutional residual blocks.

        decoder_output_mode: str
            ``"legacy_sigmoid"`` preserves the original bounded-bin decoder.
            ``"simplex_softmax"`` returns one probability mass over all bins.

        reconstruction_loss: str
            ``"mse"`` preserves the original loss. ``"forward_kl"`` computes
            target-to-reconstruction KL and requires simplex output.

        ot_loss, ot_weight: str, float
            Optional observation-space ``"sinkhorn"`` auxiliary loss and its
            non-negative weight. The Sinkhorn path requires probability-mass
            targets, simplex output, and forward KL as the base loss.

        ot_p, ot_blur, ot_scaling, ot_backend, ot_mass_epsilon:
            Strict joint Sinkhorn-divergence parameters. ``ot_support`` must be
            the normalized joint bin-center coordinates constructed from the
            fitted histogram geometry.

        condition_mode: str
            ``"none"`` disables sample-level conditioning. ``"decoder"``
            concatenates a caller-provided numeric condition vector to the
            latent vector before decoding. The encoder never receives the
            condition.

        condition_dim: int
            Width of the condition vector when ``condition_mode="decoder"``.

        """
        super().__init__()
        # check the input
        assert input_shape is not None, "!! input_shape must be given !!"
        self.dim = DataDim(len(input_shape) - 1)
        self.decoder_output_mode = validate_decoder_output_mode(
            decoder_output_mode
        )
        self.reconstruction_loss = validate_reconstruction_loss(
            reconstruction_loss
        )
        if (
                self.reconstruction_loss == "forward_kl"
                and self.decoder_output_mode != "simplex_softmax"
                ):
            raise ValueError(
                "reconstruction_loss='forward_kl' requires "
                "decoder_output_mode='simplex_softmax'."
            )
        if self.decoder_output_mode == "simplex_softmax" and input_shape[0] != 1:
            raise ValueError(
                "decoder_output_mode='simplex_softmax' currently requires "
                "a single histogram channel."
            )
        ot_config = {
            "ot_loss": ot_loss,
            "ot_weight": ot_weight,
            "ot_p": ot_p,
            "ot_blur": ot_blur,
            "ot_scaling": ot_scaling,
            "ot_backend": ot_backend,
            "ot_mass_epsilon": ot_mass_epsilon,
            "histogram_mode": (
                "probability_mass"
                if self.decoder_output_mode == "simplex_softmax"
                else None
            ),
            "decoder_output_mode": self.decoder_output_mode,
            "reconstruction_loss": self.reconstruction_loss,
        }
        validate_ot_config(ot_config)
        self.ot_loss = ot_config["ot_loss"]
        self.ot_weight = ot_config["ot_weight"]
        self.ot_p = ot_config["ot_p"]
        self.ot_blur = ot_config["ot_blur"]
        self.ot_scaling = ot_config["ot_scaling"]
        self.ot_backend = ot_config["ot_backend"]
        self.ot_mass_epsilon = ot_config["ot_mass_epsilon"]
        if self.ot_loss == "sinkhorn":
            if ot_support is None:
                raise ValueError(
                    "ot_loss='sinkhorn' requires normalized joint bin support."
                )
            self.ot_divergence = JointSinkhornDivergence(
                support=ot_support,
                p=self.ot_p,
                blur=self.ot_blur,
                scaling=self.ot_scaling,
                backend=self.ot_backend,
                mass_epsilon=self.ot_mass_epsilon,
            )
        else:
            if ot_support is not None:
                raise ValueError(
                    "ot_support must be omitted when ot_loss='none'."
                )
            self.ot_divergence = None
        self.condition_mode = validate_condition_mode(condition_mode)
        self.condition_dim = int(condition_dim)
        if self.condition_mode == "none" and self.condition_dim != 0:
            raise ValueError(
                "condition_dim must be 0 when condition_mode='none'."
            )
        if self.condition_mode == "decoder" and self.condition_dim <= 0:
            raise ValueError(
                "condition_dim must be positive when condition_mode='decoder'."
            )
        decoder_input_dim = latent_dim + self.condition_dim
        hidden_dims = hidden_dims or [32, 64, 128, 256]
        # Construct Encoder
        self.encoder = Encoder(
            input_shape[0], hidden_dims, dim=self.dim,
            dropout_conv=dropout_conv
        )
        # calculate the output shape of the encoder
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            enc_out = self.encoder(sample_input)
        self.enc_out_shape = enc_out.shape[1:]
        enc_out_dim = enc_out.numel()
        # Latent space parameters (mu, logvar)
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)
        # mapping from latent space to reconstruction features
        self.fc_decode = nn.Linear(decoder_input_dim, enc_out_dim)
        nn.init.xavier_uniform_(self.fc_decode.weight)
        nn.init.zeros_(self.fc_decode.bias)
        # Construct Decoder
        self.decoder = Decoder(
            input_shape[0], hidden_dims, dim=self.dim,
            dropout_conv=dropout_conv, output_mode=self.decoder_output_mode
        )

    def encode(self, x):
        enc_out = self.encoder(x).flatten(start_dim=1)
        mu = self.fc_mu(enc_out)
        logvar = self.fc_logvar(enc_out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _prepare_decoder_input(self, z, condition=None):
        if self.condition_mode == "none":
            if condition is not None:
                raise ValueError(
                    "condition must be omitted when condition_mode='none'."
                )
            return z
        if condition is None:
            raise ValueError(
                "condition is required when condition_mode='decoder'."
            )
        if condition.ndim != 2:
            raise ValueError(
                "condition must have shape (batch, condition_dim)."
            )
        if condition.shape[0] != z.shape[0]:
            raise ValueError("condition and latent batch sizes must match.")
        if condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"Expected condition width {self.condition_dim}, "
                f"got {condition.shape[1]}."
            )
        if not torch.isfinite(condition).all():
            raise ValueError("condition must contain only finite values.")
        condition = condition.to(device=z.device, dtype=z.dtype)
        return torch.cat([z, condition], dim=1)

    def decode(self, z, condition=None):
        decoder_input = self._prepare_decoder_input(z, condition)
        dec_input = self.fc_decode(decoder_input).view(-1, *self.enc_out_shape)
        return self.decoder(dec_input)

    def forward(self, x, sample_latent=True, condition=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        recon = self.decode(z, condition=condition)
        return recon, mu, logvar

    def _base_reconstruction_per_sample(self, recon_x, x):
        if recon_x.shape != x.shape:
            raise ValueError("Target and reconstruction shapes must match.")
        batch_size = x.size(0)
        if self.reconstruction_loss == "mse":
            return (
                (recon_x - x)
                .pow(2)
                .flatten(start_dim=1)
                .sum(dim=1)
            )

        tolerance = 1e-5
        target_flat = x.flatten(start_dim=1)
        recon_flat = recon_x.flatten(start_dim=1)
        if torch.any(target_flat < -tolerance) or torch.any(
                recon_flat < -tolerance
                ):
            raise ValueError(
                "forward_kl requires non-negative target and reconstruction."
            )
        target_sums = target_flat.sum(dim=1)
        recon_sums = recon_flat.sum(dim=1)
        ones = torch.ones(batch_size, device=x.device, dtype=x.dtype)
        if not torch.allclose(target_sums, ones, rtol=1e-5, atol=1e-5):
            raise ValueError(
                "forward_kl requires each target histogram to sum to one."
            )
        if not torch.allclose(recon_sums, ones, rtol=1e-5, atol=1e-5):
            raise ValueError(
                "forward_kl requires each reconstruction to sum to one."
            )
        eps = torch.finfo(recon_flat.dtype).eps
        elementwise = F.kl_div(
            torch.log(recon_flat.clamp_min(eps)),
            target_flat,
            reduction="none",
        )
        return elementwise.sum(dim=1)

    def observation_loss_components(self, recon_x, x, reduction="mean"):
        """Return base, Sinkhorn, weighted, and combined observation losses."""
        base = self._base_reconstruction_per_sample(recon_x, x)
        if self.ot_divergence is None:
            sinkhorn = torch.zeros_like(base)
        else:
            sinkhorn = self.ot_divergence(x, recon_x, reduction="none")
        weighted_sinkhorn = self.ot_weight * sinkhorn
        combined = base + weighted_sinkhorn

        values = {
            "base_reconstruction": base,
            "sinkhorn": sinkhorn,
            "weighted_sinkhorn": weighted_sinkhorn,
            "observation": combined,
        }
        if reduction == "none":
            return values
        if reduction == "mean":
            return {key: value.mean() for key, value in values.items()}
        if reduction == "sum":
            return {key: value.sum() for key, value in values.items()}
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")

    def vae_loss(
            self, recon_x, x, mu, logvar, beta=1.0,
            return_components=False
            ):
        """Compute observation reconstruction plus latent VAE KL."""
        batch_size = x.size(0)
        components = self.observation_loss_components(
            recon_x, x, reduction="mean"
        )
        recon_loss = components["observation"]
        # for clear understanding, we use sum instead of mean over dimensions
        kl_loss = (
            -0.5
            * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            / batch_size
        )
        total_loss = recon_loss + beta * kl_loss
        if return_components:
            return total_loss, recon_loss, kl_loss, components
        return total_loss, recon_loss, kl_loss


class LinearHead(nn.Module):
    def __init__(
            self, pretrained=None, latent_dim:int=None, num_classes:int=None, num_layers:int=2,
            hidden_head:int=512, dropout_head:float=0.3, frozen:bool=False
            ):
        """
        Parameters
        ----------
        pretrained: pre-trained model

        latent_dim: dimension of the representation

        num_classes: number of classes

        num_layers: number of layers in MLP

        hidden_head: number of hidden units in MLP
            int or list of int

        dropout_head: dropout rate

        """
        super().__init__()
        # check the input
        assert pretrained is not None, "!! pretrained model must be given !!"
        assert latent_dim is not None, "!! latent_dim must be given !!"
        assert num_classes is not None, "!! num_classes must be given !!"
        # pretrained model
        self.pretrained = pretrained
        self.frozen = frozen
        if self.frozen:
            for param in self.pretrained.parameters():
                param.requires_grad = False
        # MLP
        layers = []
        if isinstance(hidden_head, int):
            hidden_head = [hidden_head] * num_layers
        in_features = latent_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_head[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_head))
            in_features = hidden_head[i]
        layers.append(nn.Linear(hidden_head[i], num_classes))  # output layer
        self.linear_head = nn.Sequential(*layers)


    def forward(self, x, sample_latent=True, condition=None):
        mu, logvar = self.pretrained.encode(x)
        z = self.pretrained.reparameterize(mu, logvar) if sample_latent else mu
        recon = self.pretrained.decode(z, condition=condition)
        logits = self.linear_head(mu)  # use the latent representation for classification
        return logits, recon, mu, logvar


    def vae_loss(
            self, recon_x, x, mu, logvar, beta=1.0,
            return_components=False
            ):
        return self.pretrained.vae_loss(
            recon_x, x, mu, logvar, beta=beta,
            return_components=return_components
        )

    def observation_loss_components(self, recon_x, x, reduction="mean"):
        return self.pretrained.observation_loss_components(
            recon_x, x, reduction=reduction
        )
    

    def encode(self, x):
        mu, logvar = self.pretrained.encode(x)
        return mu, logvar
    

class ModelHandler:
    def __init__(self, config:dict):
        assert isinstance(config, dict), "!! config must be a dictionary !!"
        self.config = config
        self.ot_support = None

    def set_ot_support(self, support):
        self.ot_support = support

    def make_pretrain(self):
        model_params = inspect.signature(ConvVAE.__init__).parameters # diff
        model_args = {k: self.config[k] for k in model_params if k in self.config}
        if self.config.get("ot_loss", "none") == "sinkhorn":
            if self.ot_support is None:
                raise RuntimeError(
                    "Joint OT support must be prepared before model creation."
                )
            model_args["ot_support"] = self.ot_support
        model = ConvVAE(**model_args)
        for param in model.parameters():
            param.requires_grad = True
        return model
    

    def make_cpt(self, model_path):
        """
        conduct continuous pretraining

        Parameters
        ----------
        model_path: str
            path to the pretrained model
        
        """
        # load the pretrained model
        model = self.make_pretrain()
        checkpoint = torch.load(model_path)
        try:
            model.load_state_dict(checkpoint["model"])
        except KeyError:
            # if the model is saved without a dictionary
            model.load_state_dict(checkpoint)
        for param in model.parameters():
            param.requires_grad = True
        return model


    def make_finetune(self, model_path):
        """
        Parameters
        ----------
        model_path: str
            path to the pretrained model
        
        """
        # load the pretrained model
        pretrained = self.make_pretrain()
        checkpoint = torch.load(model_path)
        try:
            pretrained.load_state_dict(checkpoint["model"])
        except KeyError:
            # if the model is saved without a dictionary
            pretrained.load_state_dict(checkpoint)
        # prepare the linear head
        model_params = inspect.signature(LinearHead.__init__).parameters
        model_args = {k: self.config[k] for k in model_params if k in self.config}
        model_args["pretrained"] = pretrained
        model = LinearHead(**model_args)
        for param in model.parameters():
            param.requires_grad = True
        return model