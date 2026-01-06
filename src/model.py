import pickle
import torch.nn as nn
import torch
import os
import numpy as np
import torch.nn.functional as F

from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.conv import _ConvNd
from dynamic_network_architectures.architectures import unet
from dynamic_network_architectures.building_blocks import unet_decoder as unet_dec
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dataclasses import dataclass
from typing import Union, Type, List, Tuple, Optional
from collections import OrderedDict
from transformers.utils import ModelOutput
from samplers import SamplerWeighted, SamplerUnweighted, ConvexSamplerWeighted, PartitionedBasisSampler, SamplerMoreProper, ProperPartionedSampler, PerVoxelBasisSampler, WrappedTorchDist

@dataclass
class UncertaintyModelOutput(ModelOutput):
    mu: Optional[torch.FloatTensor] = None
    cov_mat: Optional[torch.FloatTensor] = None
    sigma_diag: Optional[torch.FloatTensor] = None
    sample_logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    loss_decomp: Optional[dict] = None


@dataclass 
class UnetDecoderOutput(ModelOutput):
    mu: Optional[torch.FloatTensor] = None
    cov_mat: Optional[torch.FloatTensor] = None
    sigma_diag: Optional[torch.FloatTensor] = None
    cov_weighting: Optional[torch.FloatTensor] = None
    cov_weighting_soft: Optional[torch.FloatTensor] = None
    ppt_mat: Optional[torch.FloatTensor] = None


class ConstantScheduler:

    def __init__(self, initial_tau, *args, **kwargs):
        
        self.initial_tau = initial_tau
    
    def step(self):
        return self.initial_tau
    def get_tau(self):
        return self.initial_tau
    def reset(self):
        return None


class LinearTauScheduler:
    def __init__(self, initial_tau: float, min_tau: float, total_steps: int):
        assert initial_tau > 0, "initial_tau must be > 0"
        assert min_tau > 0, "min_tau must be > 0"
        assert initial_tau >= min_tau, "initial_tau must be >= min_tau"
        assert total_steps > 0, "total_steps must be > 0"
        
        self.initial_tau = initial_tau
        self.min_tau = min_tau
        self.total_steps = total_steps
        self.current_step = 0

        # Precompute slope for efficiency
        self.slope = (initial_tau - min_tau) / total_steps

    def step(self):
        """Advance one step and return the new tau."""
        tau = max(
            self.min_tau,
            self.initial_tau - self.slope * self.current_step
        )
        self.current_step += 1
        return tau

    def get_tau(self):
        """Return current tau without stepping."""
        tau = max(
            self.min_tau,
            self.initial_tau - self.slope * self.current_step
        )
        return tau

    def reset(self):
        """Reset the scheduler to start over."""
        self.current_step = 0



class LowRankLogitCovHead3D(nn.Module):
    def __init__(self, in_ch, n_classes, rank=4, hidden=0, eps=1e-4):
        super().__init__()
        self.C = n_classes
        self.r = rank
        self.eps = eps

        if hidden > 0:
            self.pre = nn.Sequential(
                nn.Conv3d(in_ch, hidden, 1, bias=True),
                nn.GroupNorm(8, hidden),
                nn.SiLU(),
            )
            feat_ch = hidden
        else:
            self.pre = nn.Identity()
            feat_ch = in_ch

        self.mu_head   = nn.Conv3d(feat_ch, n_classes, 1)   # μ(x)
        self.coeff_head= nn.Conv3d(feat_ch, rank, 1)        # a(x)
        self.diag_head = nn.Conv3d(feat_ch, n_classes, 1)   # s(x) -> σ=softplus(s)+eps

        B_init = torch.randn(n_classes, rank) * 0.02 if rank > 0 else torch.zeros(n_classes, 0)
        self.B = nn.Parameter(B_init, requires_grad=(rank > 0))

    def forward(self, feats):
        feats = self.pre(feats)
        mu = self.mu_head(feats)                       # [B,C,D,H,W]
        if self.r > 0:
            a  = self.coeff_head(feats)               # [B,r,D,H,W]
        else:
            a = torch.zeros(mu.size(0), 0, *mu.shape[2:], device=mu.device, dtype=mu.dtype)
        s  = self.diag_head(feats)                     # [B,C,D,H,W]
        sigma = F.softplus(s) + self.eps
        return mu, a, sigma, self.B


class CovarianceWeightHead(nn.Module):

    def __init__(self, in_channels, num_bases, use_global_pool = True):
        super().__init__()
        """
        in_channels: number of channels in the input features (from U-Net backbone)
        num_bases: N, number of global covariance basis matrices
        use_global_pool: if True, outputs one weight vector per volume; 
                         if False, outputs one weight vector per voxel
        """

        super().__init__()
        self.num_bases = num_bases
        self.use_global_pool = use_global_pool

        # small convolutional head
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels // 4, num_bases, kernel_size=1)  # 
        
        self.chosen_indices = []
    
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        logits = self.conv3(h)

        if self.use_global_pool:
            # global average pooling over spatial dims -> one weight vector per volume
            logits = logits.mean(dim=(2,3,4))  # (B, N)
        # else, keep voxel-wise weights: (B, N, D, H, W)

        # softmax to get convex combination weights
        weights = F.softmax(logits, dim=1)
        return weights
    

class PerVoxelCovarianceHead(nn.Module):

    def __init__(self, in_channels, num_bases):
        """
        Class to compute the per pixel weigting of the covariance bases.
        """
        super().__init__()

        self.weight_head = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),  # smoother than ReLU
            nn.Conv3d(in_channels, num_bases, kernel_size=1)
        )
        self.in_channels = in_channels
        self.num_bases = num_bases
        self.softmax = nn.Softmax(dim = 1)
        self._track_chosen_indices = False
        self._basis_counts = {i: 0 for i in range(self.num_bases)}
        self.total_numel = 0

    def track_chosen_indices(self, track = True):
        self._track_chosen_indices = track
        self._basis_counts = {i: 0 for i in range(self.num_bases)}
        self.total_numel = 0
    
    @property
    def basis_counts(self):
        return {key: val / self.total_numel for key, val in self._basis_counts.items()}

    def forward(self, x):
        out = self.weight_head(x)
        weights = self.softmax(out)

        if self._track_chosen_indices:
            chosen = weights.argmax(1).view(-1)
            for i in range(self.num_bases):
                self._basis_counts[i] += torch.sum(chosen == i)
                self.total_numel += weights.numel()

        return weights, weights
        

class CovarianceWeightHeadLight(nn.Module):
    """
    Predicts per-volume or per-voxel weights for N covariance bases.
    Fully convolutional. Maps input channels to N bases, then optionally reduces spatially.
    """
    def __init__(self, in_channels, num_bases, use_global_pool=True, top_k = 1):
        """
        in_channels: number of channels in the feature map (e.g., 32)
        num_bases: N covariance bases
        use_global_pool: True -> one weight vector per volume, False -> voxel-wise
        """
        super().__init__()
        self.num_bases = num_bases
        self.use_global_pool = use_global_pool

        # channel-wise linear projection
        self.conv = nn.Conv3d(in_channels, num_bases, kernel_size=1)
        self.top_k = top_k
        self.chosen_indices = []
        self._track_chosen_indices = False

    def track_chosen_indices(self, track = True):
        self._track_chosen_indices = track
        self.chosen_indices = []
    
    def forward(self, x, tau = 0.9):
        """
        x: [B, in_channels, D, H, W]
        Returns:
            weights: [B, N] if use_global_pool=True, else [B, N, D, H, W]
        """
        logits = self.conv(x)  # [B, N, D, H, W]

        if self.use_global_pool:
            # global average pooling over spatial dims -> one weight vector per volume
            logits = logits.mean(dim=(2,3,4))  # [B, N]
        
        weights = F.gumbel_softmax(logits, tau, dim=1, hard = True if self.top_k == 1 else False)

        if self._track_chosen_indices:
            self.chosen_indices.extend(list(torch.argmax(weights, axis = -1).detach().cpu().int()))
        
        return weights

class PartionedCovHead(nn.Module):

    def __init__(self, C, N):
        super().__init__()
        # Selector network
        self.selector = nn.Conv3d(C, N, kernel_size=1)        
        self.num_parts = 4
        self._track_chosen_indices = False
        self.basis_counts = None
        self.chosen_indices = []
        self.linear = nn.Linear(N*self.num_parts**3, N*self.num_parts**3)
        self.tau = 0.3
    def track_chosen_indices(self, track = True):
        self._track_chosen_indices = track
        self.basis_counts = None
    
    def gumbel_softmax_sample(self, logits, tau,dim = 1, eps = 1e-20):
        """
        Draw a sample from the Gumbel-Softmax distribution (soft)
        Args:
            logits: [B, G, N] unnormalized log-probs
            tau: temperature
        Returns:
            y: [B, G, N] soft probabilities summing to 1
        """
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y = F.softmax((logits + gumbel_noise) / tau, dim=dim)
        return y

    def gumbel_softmax(self, logits, tau, dim = 1, hard = True):
        """
        Sample from Gumbel-Softmax and optionally discretize to hard one-hot
        Args:
            logits: [B, G, N] unnormalized log-probs
            tau: temperature
            hard: if True, returns one-hot vector but keeps gradient
        Returns:
            y_hard: [B, G, N] hard one-hot (or soft if hard=False)
            y_soft: [B, G, N] soft probabilities
        """
        y_soft = self.gumbel_softmax_sample(logits, tau=tau, dim = dim)

        if hard:
            # Straight-through trick
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            # gradient flows through y_soft
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft

        return y, y_soft

    def forward(self, x):
        
        # ---- 1) Select basis per partition ----
        logits = self.selector(x)  # [B, N, H, W, D]
        logits = F.adaptive_avg_pool3d(logits, (self.num_parts,)*3)
        logits = nn.GELU()(logits)
        B, N, np, _, _ = logits.shape

        logits = self.linear(logits.view(B, -1)).view(B, N, np, np, np)
        #y_hard, y_soft = self.gumbel_softmax(logits, tau, dim=1, hard = self.hard)                         # [B, N, np, np, np]

        y_hard = F.softmax(logits / self.tau, dim = 1) 
        y_soft = y_hard 

        if self._track_chosen_indices:
            # indices: [B, np, np, np], values in [0, N-1]
            indices = torch.argmax(y_hard, dim=1).detach().cpu()

            # Flatten and count how many times each basis was used
            counts = torch.bincount(indices.flatten(), minlength=y_hard.size(1))

            # Option 1: accumulate counts over training
            if self.basis_counts is None:
                self.basis_counts = torch.zeros(y_hard.size(1), dtype=torch.long)
            self.basis_counts += counts

        return y_hard, y_soft

class UnetWithUncertainty(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Union[None, dict] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Union[None, dict] = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: Union[None, dict] = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
        cov_rank: int = 16,
        num_samples_train: int = 5,
        num_samples_inference: int = 30, 
        sample_type: str = 'ours', 
        return_samples: bool = True,
        loss_kwargs: Union[dict, None] = None,
        predict_cov_weighting = False,
        cov_weighting_kwargs = None,
        ppt_kwargs = None,
        **kwargs
    ):
        super().__init__()

    
        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stem"
        self.keys_to_in_proj = ("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0")

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, (
            "n_blocks_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_blocks_per_stage: {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            disable_default_stem=False,
            stem_channels=stem_channels,
        )

        self.decoder = UnetDecoderWithUncertainty(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, cov_rank=cov_rank, num_samples=num_samples_train, predict_cov_weighting=predict_cov_weighting, cov_weighting_kwargs=cov_weighting_kwargs, ppt_kwargs=ppt_kwargs)
        B_init = torch.randn(num_classes, cov_rank) * 0.02 if cov_rank > 0 else torch.zeros(num_classes, 0)
        self.cov_basis_mat = nn.Parameter(B_init, requires_grad=(cov_rank > 0))

        if loss_kwargs is None:
            loss_kwargs = {}
        
        
        self.loss_fn = VariationalLoss(**loss_kwargs)
        self.num_samples_train = num_samples_train
        self.num_samples_inference = num_samples_inference

        self.return_samples = return_samples
        self.cov_rank = cov_rank
        self.min_logvar, self.max_logvar = -5, 5
        self.evaluate_with_samples = True
        self.sample_type = sample_type
        self.sigmoid = nn.Sigmoid()
        self.link_function = nn.Softmax(dim = 1)


    def forward(self, x, targets = None, reduction = "none", scaler = None):

        if scaler is not None:
            output = self.online_sampling_and_loss_gradient_accum(x, targets, scaler)
            return output
    
        skips = self.encoder(x)
        output = self.decoder(skips)

        mu, log_var_diag, cov_out, weighting, weighting_soft, ppt_param = (
            output.mu, output.sigma_diag, output.cov_mat, output.cov_weighting, output.cov_weighting_soft, output.ppt_mat
        )

        diag_var_out = torch.exp(0.5 * torch.clamp(log_var_diag, self.min_logvar, self.max_logvar))
        if self.decoder.deep_supervision:
            mu = mu[0]

        logits = None
        loss, loss_attributes = None, None

        if not self.evaluate_with_samples:
            if targets is not None:
                loss, loss_attributes = self.loss_fn(mu, targets)
            return UncertaintyModelOutput(self.link_function(mu), cov_out, diag_var_out, logits, loss, loss_attributes)
        
        num_samples = self.num_samples_train if self.training else self.num_samples_inference
        if targets is not None:
            loss, loss_attributes, mu =  self.online_sampling_and_loss(targets, mu, cov_out, diag_var_out, weighting=weighting, soft_weighting = weighting_soft, num_samples=num_samples)
        else:
            mu = self.online_sampling(mu, cov_out, diag_var_out, num_samples=num_samples)
            
        output = UncertaintyModelOutput(mu, cov_out, diag_var_out, logits, loss, loss_attributes)
        return output
    
    def online_sampling_and_loss_gradient_accum(self, x, targets, scaler):

        for idx in range(self.num_samples_train):
            skips = self.encoder(x)
            mu, log_var_diag, cov_out, weighting = self.decoder(skips)

            diag_var_out = torch.exp(0.5 * torch.clamp(log_var_diag, self.min_logvar, self.max_logvar))
            if self.decoder.deep_supervision:
                mu = mu[0]
            
            num_samples = 1
            loss, loss_attributes, prediction = self.online_sampling_and_loss(targets, mu, cov_out, diag_var_out,weighting=weighting, num_samples=num_samples)
            scaler.scale(loss / self.num_samples_train).backward()

        output = UncertaintyModelOutput(mu, cov_out, diag_var_out, prediction, loss, loss_attributes)
        return output
            
    def basis_model_only(self, only = True):
        self.evaluate_with_samples = not only

    def online_sampling_and_loss(self, targets, mu, a, sigma, weighting, soft_weighting, num_samples, scaler = None):
        
        loss, loss_attributes = 0., {}

        distribution = self.get_distribution(mu, a, sigma, self.cov_basis_mat, weighting)
        prediction = None
        for idx in range(num_samples):
            sample = distribution.sample()
            sample = sample.view(mu.shape)
            if idx == 0:
                prediction = F.softmax(sample.detach(), dim = 1)
                
                if isinstance(distribution, (ConvexSamplerWeighted, )):
                    comb_loss, sample_attributes = self.loss_fn(sample, targets, torch.log(sigma), weighting=soft_weighting, cov_basis = self.cov_basis_mat)
                else:
                    comb_loss, sample_attributes = self.loss_fn(sample, targets, torch.log(sigma), weighting = soft_weighting, cov_basis = self.cov_basis_mat)

                loss_attributes = {key: 0 for key in sample_attributes.keys()}
            else:
                comb_loss, sample_attributes = self.loss_fn(sample, targets)
                prediction += F.softmax(sample.detach(), dim = 1)
               
            loss += comb_loss
            loss_attributes = {key: val + sample_attributes[key] for key, val in loss_attributes.items()}

        loss_attributes = {key: val / num_samples for key, val in loss_attributes.items()}    
        prediction = prediction / num_samples

        return loss, loss_attributes, prediction
    

    def get_distribution(self, mu, low_rank, sigma, basis_matrix, weighting):
        
        if self.sample_type == 'torch':
            return self.get_torch_distribution(mu, low_rank, sigma, basis_matrix)
        
        elif self.sample_type in ['ours', 'unweighted']:
            return SamplerUnweighted(mu, sigma, low_rank_cov= low_rank, cov_basis_mat=basis_matrix, weighting=weighting)
        elif self.sample_type  == 'weighted':
            return SamplerWeighted(mu, sigma, low_rank_cov= low_rank, cov_basis_mat=basis_matrix, weighting=weighting)
        elif self.sample_type == 'convex':
            return ConvexSamplerWeighted(mu, sigma, low_rank_cov= low_rank, cov_basis_mat=basis_matrix, weighting=weighting)
        elif self.sample_type == 'partitioned':
            return PartitionedBasisSampler(mu, sigma, low_rank, basis_matrix, weighting=weighting)
        elif self.sample_type == 'diagonal':
            return self.get_independent_torch_distribution(mu, low_rank, sigma, basis_matrix)
        elif self.sample_type == 'basic_proper':
            return SamplerMoreProper(mu, sigma, low_rank, basis_matrix, weighting=weighting)
        elif self.sample_type == 'partitioned_proper':
            return ProperPartionedSampler(mu, sigma, low_rank, basis_matrix, weighting=weighting)
        elif self.sample_type == 'per_voxel_multi_basis':
            return PerVoxelBasisSampler(mu, sigma, low_rank, basis_matrix, weighting = weighting)
        else:
            raise NotImplementedError(f"Unknown sampler: {self.sample_type}")

    def online_sampling(self, mu, a, sigma,weighting, num_samples):
        
        prediction = torch.zeros_like(mu)
        distribution = self.get_distribution(mu, a, sigma, self.cov_basis_mat, weighting=weighting)
        
        for idx in range(num_samples):
            prediction += distribution.sample().view(prediction.shape)
        return prediction / num_samples
    

    def get_torch_distribution(self, mu, a, sigma, B):
        batch_size, C, D, H, W = mu.shape
        # a = a.transpose(1,2).view(batch_size, self.cov_rank // 2, -1).transpose(1,2)
    
        # dist = torch.distributions.LowRankMultivariateNormal(
        #     mu.view(batch_size, -1), a.permute(0,2,3,4,1).reshape(batch_size, D*H*W, self.cov_rank), sigma.view(batch_size, -1)
        # )

        
        
        dist = torch.distributions.LowRankMultivariateNormal(
            mu.view(batch_size, -1), a.reshape(batch_size, C*D*H*W, self.cov_rank // C), sigma.view(batch_size, -1)
        )
        
        dist = WrappedTorchDist(dist)

        return dist
    
    def get_independent_torch_distribution(self, mu, a, sigma, B):

        batch_size, C, D, H, W = mu.shape
        dist = torch.distributions.Normal(mu.view(batch_size, -1), sigma.view(batch_size, -1))
        return dist
                
    def calculate_loss(self, targets, logits_mc, sigma, is_sampled = True):
        
        loss, loss_attributes = 0., {}
        if not is_sampled:
            comb_loss, loss_attributes = self.loss_fn(logits_mc, targets, torch.log(sigma))
            return comb_loss, loss_attributes
    
        for idx in range(self.num_samples):
            if idx == 0:
                comb_loss, sample_attributes = self.loss_fn(logits_mc[idx], targets, torch.log(sigma))
                loss_attributes = {key: 0 for key in sample_attributes.keys()}
            else:
                comb_loss, sample_attributes = self.loss_fn(logits_mc[idx], targets)  # Only KL loss once

            loss += comb_loss
            loss_attributes = {key: val + sample_attributes[key] for key, val in loss_attributes.items()}
    
        return loss, loss_attributes


class UnetDecoderWithUncertainty(unet_dec.UNetDecoder):
    
    def __init__(self, 
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: Union[None, dict] = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: Union[None, dict] = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: Union[None, dict] = None,
                 conv_bias: Union[None, bool] = None,
                 cov_rank: int = 16,
                 num_samples: int = 5,
                 predict_cov_weighting = False,
                 cov_weighting_kwargs = None,
                 ppt_kwargs = None,
                 **loss_kwargs
                 ):

        super().__init__(encoder, num_classes, n_conv_per_stage, deep_supervision, nonlin_first, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, conv_bias)

        self.cov_rank = cov_rank
        self.cov_weighting_kwargs = cov_weighting_kwargs if isinstance(cov_weighting_kwargs, dict) else {}
        self.ppt_kwargs = ppt_kwargs if isinstance(ppt_kwargs, dict) else {} 
        self.predict_cov_weighting = predict_cov_weighting
        self.output_stage_meta_diag = {
            'in_channels': self.seg_layers[-1].in_channels,
            'out_channels': self.seg_layers[-1].out_channels,
            'kernel_size': self.seg_layers[-1].kernel_size,
            'stride': self.seg_layers[-1].stride,
            'padding': self.seg_layers[-1].padding
        }

        self.output_stage_meta_cov = self.output_stage_meta_diag.copy()
        self.output_stage_meta_cov['out_channels'] = cov_rank

        self.diag_head = self.encoder.conv_op(**self.output_stage_meta_diag)
        self.cov_head = self.encoder.conv_op(**self.output_stage_meta_cov)
        
        if self.predict_cov_weighting:
        
            num_bases = cov_weighting_kwargs.get('num_bases', cov_rank)

            covariance_head_type = cov_weighting_kwargs.get('class', CovarianceWeightHeadLight)
            
            self.cov_weighting_head = covariance_head_type(
                self.diag_head.in_channels, num_bases
            )
        
    
        self.combine_methods = ppt_kwargs.get('include', False)
        self.ppt_rank = None
        if self.combine_methods:
            self.ppt_rank = ppt_kwargs.get('rank', self.cov_rank)
            ppt_output_state_meta = self.output_stage_meta_cov.copy()
            ppt_output_state_meta['out_channels'] = self.ppt_rank
            self.ppt_head = self.encoder.conv_op(**ppt_output_state_meta)

        self.return_samples = True
        self.num_samples = num_samples


    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :param targets:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        diag_var_out, cov_out, cov_weighting, cov_weighting_soft, ppt_param = None, None, None, None, None
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)

            is_last = s == len(self.stages) - 1
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
                if is_last:
                    diag_var_out = self.diag_head(x)
                    cov_out = self.cov_head(x)
                    if self.predict_cov_weighting:
                        cov_weighting, cov_weighting_soft = self.cov_weighting_head(x)

            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
                diag_var_out = self.diag_head(x)
                cov_out = self.cov_head(x)
                if self.predict_cov_weighting:
                    cov_weighting, cov_weighting_soft = self.cov_weighting_head(x)
                
                if self.combine_methods:
                    ppt_param = self.ppt_head(x)
            lres_input = x
    
        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]
        seg_outputs = seg_outputs[0] if not self.deep_supervision else seg_outputs
        output = UnetDecoderOutput(seg_outputs,cov_out, diag_var_out, cov_weighting, cov_weighting_soft, ppt_param)
        return output
        

class MultipleBasisUNetWithUncertainty(UnetWithUncertainty):

    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias = False, norm_op = None, norm_op_kwargs = None, dropout_op = None, dropout_op_kwargs = None, nonlin = None, nonlin_kwargs = None, deep_supervision = False, block = BasicBlockD, bottleneck_channels = None, stem_channels = None, cov_rank = 16, num_samples_train = 5,num_samples_inference = 30, return_samples = True, loss_kwargs = None,cov_weighting_kwargs = None, sample_type = 'ours', **kwargs):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, block, bottleneck_channels, stem_channels, cov_rank, num_samples_train,num_samples_inference, sample_type, return_samples, loss_kwargs,predict_cov_weighting = True, cov_weighting_kwargs = cov_weighting_kwargs, **kwargs)
        
        self.num_bases = cov_weighting_kwargs.get('num_bases', 5)
        self.cov_basis_mat = nn.Parameter(torch.randn(self.num_bases, num_classes, cov_rank))
        self.sample_type = cov_weighting_kwargs.get('sample_type', 'weighted')

        if self.sample_type == 'convex':
            self.decoder.cov_weighting_head.top_k = self.num_bases
        
        

    
    
class PPUnetWithUncertainty(UnetWithUncertainty):

    def __init__(self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Union[None, dict] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Union[None, dict] = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: Union[None, dict] = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
        cov_rank: int = 16,
        num_samples: int = 5,
        return_samples: bool = True,
        loss_kwargs: Union[dict, None] = None,
        **kwargs):
        args = (input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage, 
                num_classes, n_blocks_per_stage, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, deep_supervision, block, bottleneck_channels, stem_channels, cov_rank, num_samples, return_samples, loss_kwargs)
        
        super().__init__(*args, **kwargs)


        del self.cov_basis_mat
    

    def forward(self, x, target = None, reduction = 'mean'):
        skips = self.encoder(x)
        mu, log_var_diag, cov_out = self.decoder(skips)

        diag_var_out = torch.exp(0.5 * torch.clamp(log_var_diag, self.min_logvar, self.max_logvar))

        if self.decoder.deep_supervision:
            mu = mu[0]

        logits = None
        loss, loss_attributes = None, None
        if self.return_samples and self.evaluate_with_samples:
            logits_mc = self.sample_logits(
                mu, cov_out, diag_var_out, self.cov_basis_mat, num_samples=self.num_samples
            )

            if targets is not None:
                is_sampled = True
                if reduction == 'mean':
                    logits_mc = logits_mc.mean(0)
                    is_sampled = False
                
                loss, loss_attributes = self.calculate_loss(targets, logits_mc, diag_var_out, is_sampled=is_sampled)
        else:
            if targets is not None:
                loss, loss_attributes = self.calculate_loss(targets, mu, diag_var_out, is_sampled=False)

        output = UncertaintyModelOutput(mu, cov_out, diag_var_out, logits, loss, loss_attributes)
        return output

    def sample_logits(self, mu, diag_var_out, cov_out, num_samples):
        """
        Sample MC logits from a Gaussian with low-rank + diagonal covariance.

        Distribution:
            logits ~ N(mu, Σ), with
            Σ = (P)(P)^T + diag(sigma^2)

        Args:
            mu:    [B,C,D,H,W]   mean logits
            cov_out:     [B,r,D,H,W]   low-rank scales (voxel dependent)
            sigma: [B,C,D,H,W]   diagonal std dev
            num_samples: int     number of Monte Carlo samples
            targets: [B,D,H,W]   (optional) target labels, unused here but kept for API compatibility

        Returns:
            logits_mc: [K,B,C,D,H,W]   sampled logits
        """


        batch_size, C, D, H, W = mu.shape
        z_one = torch.randn(num_samples, batch_size, C, D, H, W, device=mu.device, dtype=mu.dtype)
        z_two = torch.randn(num_samples, batch_size, self.cov_rank, device = mu.device, dtype = mu.dtype)

        scalar = 1 / (torch.sqrt(2*(self.cov_rank -1))) 

        low_rank = scalar * cov_out.permute(0, 2, 3, 4, 1).view(batch_size, -1, self.cov_rank) 




        



class UncertaintySegLoss(nn.Module):
    def __init__(self,
                 lambda_ce=1.0,
                 lambda_dice=0.0,
                 lambda_nll=1.0,
                 lambda_kl=1e-4,
                 smooth=1e-6,
                 reduction="mean"):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.lambda_nll = lambda_nll
        self.lambda_kl = lambda_kl
        self.smooth = smooth
        self.reduction = reduction

    def dice_loss(self, probs, target_onehot):
        """
        probs: (B, C, D, H, W) softmax predictions
        target_onehot: (B, C, D, H, W) one-hot ground truth
        """
        dims = tuple(range(2, probs.ndim))  # spatial dims
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs**2 + target_onehot**2, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1. - dice_score.mean()
        return dice_loss

    def forward(self, mu_logits, log_var, target, **kwargs):
        """
        mu_logits: (B, C, D, H, W) predicted mean logits
        log_var:   (B, C, D, H, W) predicted log-variances
        target:    (B, D, H, W) integer class labels
        """
        B, C, *spatial = mu_logits.shape
        N = B * torch.prod(torch.tensor(spatial, device=mu_logits.device))

        # ---- Cross Entropy Loss ----
        ce_loss = F.cross_entropy(mu_logits, target, reduction=self.reduction)

        # ---- Dice Loss ----
        probs = F.softmax(mu_logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=C).permute(0, -1, *range(1, len(spatial)+1)).float()
        dice_loss = self.dice_loss(probs, target_onehot)

        # ---- Negative Log Likelihood (Gaussian over logits) ----
        var = torch.exp(log_var)
        precision = 1.0 / var
        nll = 0.5 * torch.sum(
            (target_onehot - mu_logits)**2 * precision + log_var
        ) / N

        # ---- KL regularization on log-variance ----
        kl = 0.5 * torch.sum(var - 1.0 - log_var) / N

        # ---- Combine ----
        loss = (self.lambda_ce * ce_loss +
                self.lambda_dice * dice_loss +
                self.lambda_nll * nll +
                self.lambda_kl * kl)

        return loss, {
            "ce": ce_loss.item(),
            "dice": dice_loss.item(),
            "nll": nll.item(),
            "kl": kl.item()
        }


class VariationalLoss(nn.Module):
    def __init__(self,
                 lambda_ce=1.0,
                 lambda_dice=0.0,
                 lambda_nll=1.0,
                 lambda_kl=1e-4,
                 smooth=1e-6,
                 lambda_weight = 0.05,
                 lambda_orth = 1,
                 reduction="mean",
                 num_total = 100,
                 **kwargs):

        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.lambda_nll = lambda_nll
        self.lambda_kl = lambda_kl
        self.smooth = smooth
        self.reduction = reduction
        self.ce_forward = nn.CrossEntropyLoss(reduction=reduction)
        self.num_total = num_total
        self.lambda_orth = lambda_orth
        self.lambda_weight = lambda_weight

        print("Loss initialized with ce:", lambda_ce, 'dice:', lambda_dice, 'kl:', lambda_kl)
    
    def forward(self, logits, target, log_var = None, weighting = None, cov_basis = None):
        
        ce_loss = self.ce_forward(logits, target)
        orth_loss, weighting_loss = torch.zeros_like(ce_loss),torch.zeros_like(ce_loss) 

        dice_loss = self.dice_loss(F.softmax(logits, dim = 1), target, num_classes = logits.shape[1])
        kl_loss = torch.zeros_like(ce_loss) if log_var is None else self.kl_loss(log_var, self.num_total)

        loss = self.lambda_ce * ce_loss + self.lambda_dice * dice_loss + self.lambda_kl * kl_loss

        if weighting is not None:
            weighting_loss = self.weighting_loss(weighting) 
            loss += weighting_loss * self.lambda_weight
        if cov_basis is not None:
            orth_loss = self.orthogonal_cov_basis_loss(cov_basis)
            loss += orth_loss * self.lambda_orth
        
        return loss, {'cross_entropy': ce_loss.item(), 'dice': dice_loss.item(), 'kl': kl_loss.item(),
                      'orth': orth_loss.item(), 'weight': weighting_loss.item()}

    @staticmethod
    def weighting_loss(weighting):
        """
        param weighting: torch.tensor of shape (N, R) i.e the softmax outputs of the weighting of the bases 
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        entropy = - (weighting * (weighting + eps).log()).sum(dim=1)  

        # Mean entropy across batch & volume
        entropy_loss = - entropy.mean()
        return entropy_loss

    
    @staticmethod
    def orthogonal_cov_basis_loss(cov_basis):
        """
        param cov_basis: torch.tensor of shape (N, C, R) basis matrices for the covariance  
        """

        N = cov_basis.shape[0]
        B_flat = cov_basis.view(N, -1)  # [N, C*r]

        # Normalize rows (optional, for scale invariance)
        B_norm = F.normalize(B_flat, dim=1)

        # Compute Gram matrix
        G = torch.matmul(B_norm, B_norm.T)  # [N, N]

        # Penalize off-diagonal entries
        I = torch.eye(G.shape[0], device=G.device)
        orth_loss = ((G - I)**2).sum() / (N * (N - 1))
        return orth_loss

    @staticmethod
    def kl_loss(log_var, num_total = 1000):
        """
        Calculates kl div loss in approximate setting, log var is only diagonal to reduce computation
        :param log_var: predicted variances
        :param num_total: the total number of samples in training, so to act like a prior
        :return: KL Div
        """
        num_total = log_var.numel()
        return 0.5 * torch.sum(torch.exp(log_var) - 1 - log_var) / num_total

    @staticmethod
    def dice_loss(probs, target, num_classes, eps=1e-6):
        target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersect = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)
        dice = (2. * intersect / (cardinality + eps)).mean()
        return 1. - dice


MODEL_CLASSES = {
    'standard_uncertainty': UnetWithUncertainty,
    'weighted_basis': MultipleBasisUNetWithUncertainty
}

def get_model_from_base_kwargs(path_to_base, **kwargs):

    base_kwargs = pickle.load(open(path_to_base, 'rb'))
    base_kwargs['loss_kwargs'] = kwargs.pop('loss_kwargs', dict())
    base_kwargs['ppt_kwargs'] = kwargs.pop('ppt_kwargs', dict())
    
    model_class_name = kwargs.pop('model_type', 'standard_uncertainty')
    print(model_class_name)
    print(kwargs)
    model_class = MODEL_CLASSES[model_class_name]
    for key, val in kwargs.items():
        base_kwargs[key] = val

    model = model_class(**base_kwargs)
    return model



if __name__ == '__main__':

    model_kwargs = {
        'checkpoint_path': '/scratch/pjtka/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        'path_to_base': '/scratch/awias/data/nnUNet/info_dict_TotalSegmentatorPancreas.pkl',
        'model_type': 'weighted_basis',
        'cov_weighting_kwargs': {
            'num_bases': 5
        } 

    }
    
    path_to_base = r"/home/pjtka/ndsegment/NDSegRef/uncertainty/info_dict.pkl"
    path_to_base = model_kwargs.pop('path_to_base')
    
    model = get_model_from_base_kwargs(path_to_base, **model_kwargs)
    model = model.to('cuda')
    data = torch.rand((2, 1, 128, 128, 128)).to('cuda')
    model(data)
    breakpoint()
    state_dict = torch.load(
        '/scratch/pjtka/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth'
    ,weights_only=False)
    own_state_dict = model.state_dict()
    breakpoint()
    model.load_state_dict(state_dict, strict=True)
    
    data = torch.rand((2, 1, 224, 224, 224))
    # out = model(data.to('cuda'))
    targets = (torch.randn((2, 1, 224, 224, 224)) > 0).to('cuda').long()
    out = model(data.to('cuda'), targets)
    breakpoint()
















