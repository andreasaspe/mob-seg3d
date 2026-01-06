import torch.nn as nn
import torch 
import torch.nn.functional as F


class WrappedTorchDist(nn.Module):
    def __init__(self, torch_dist):
        self.torch_dist = torch_dist

    def sample(self, ):
        return self.forward()

    def forward(self, ):
        return self.torch_dist.rsample()
    

class BaseSampler(nn.Module):

    def __init__(self,mu, sigma, low_rank_cov, cov_basis_mat, weighting):
        super().__init__()

        self.mu = mu 
        self.sigma = sigma
        self.low_rank_cov = low_rank_cov
        self.cov_basis_mat = cov_basis_mat
        self.weighting = weighting
    
    def sample(self, size = 1):
        
        if size > 1:
            samples = torch.cat([self.foward().unsqueeze(0) for _ in range(size)], dim = 0)
            return samples
        return self.forward()

    def forward(self,):
        pass


class SamplerMoreProper(BaseSampler):

    def __init__(self, mu, sigma, low_rank_cov, cov_basis_mat, weighting):
        super().__init__(mu, sigma, low_rank_cov, cov_basis_mat, weighting)
    
    def forward(self):
        
        """
        Sample MC logits from a Gaussian with low-rank + diagonal covariance.

        Distribution:
            logits ~ N(mu, Σ), with
            Σ = (B diag(a))(B diag(a))^T + diag(sigma^2)

        Args:
            mu:    [B,C,D,H,W]   mean logits
            a:     [B,r,D,H,W]   low-rank scales (voxel dependent)
            sigma: [B,C,D,H,W]   diagonal std dev
            B:     [C,r]         learned basis matrix (shared across voxels/batch)
            num_samples: int     number of Monte Carlo samples
            targets: [B,D,H,W]   (optional) target labels, unused here but kept for API compatibility

        Returns:
            logits_mc: [K,B,C,D,H,W]   sampled logits
        """
        Bsz, C, D, H, W = self.mu.shape
        r = self.low_rank_cov.shape[1]  # rank

        # -------------------------------------------------------
        # 1. Expand mean for sampling
        # -------------------------------------------------------
        mu_exp = self.mu.unsqueeze(0)  # [K,B,C,D,H,W]

        # -------------------------------------------------------
        # 2. Low-rank noise component
        # -------------------------------------------------------
        # eps_r ~ N(0, I), shape [K,B,r,D,H,W]
        eps_r = torch.randn(1, Bsz, r, device=self.mu.device, dtype=self.mu.dtype)

        # Scale by 'a': [K,B,r,D,H,W]
        scaled = eps_r.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.low_rank_cov.unsqueeze(0)

        # Reshape to [K,B,r,N] with N = D*H*W
        N = D * H * W
        scaled = scaled.view(1, Bsz, r, N)

        # Project into C with B:
        #   B: [C,r], scaled: [K,B,r,N]
        #   result: [K,B,C,N]
        low_rank = torch.matmul(self.cov_basis_mat, scaled)  

        # Reshape back to [K,B,C,D,H,W]
        low_rank_noise = low_rank.view(1, Bsz, C, D, H, W)

        # -------------------------------------------------------
        # 3. Diagonal noise component
        # -------------------------------------------------------
        eps_diag = torch.randn(1, Bsz, C, D, H, W, device=self.mu.device, dtype=self.mu.dtype)
        diag_noise = eps_diag * self.sigma.unsqueeze(0)

        # -------------------------------------------------------
        # 4. Combine all parts
        # -------------------------------------------------------
        logits_mc = mu_exp + low_rank_noise + diag_noise
        return logits_mc.squeeze(0)


class SamplerUnweighted(BaseSampler):

    def __init__(self, mu, sigma, low_rank_cov, cov_basis_mat, weighting):
        super().__init__(mu, sigma, low_rank_cov, cov_basis_mat, weighting)

    
    def forward(self, ):

        """
        Sample MC logits from a Gaussian with low-rank + diagonal covariance.

        Distribution:
            logits ~ N(mu, Σ), with
            Σ = (B diag(a))(B diag(a))^T + diag(sigma^2)

        Args:
            mu:    [B,C,D,H,W]   mean logits
            a:     [B,r,D,H,W]   low-rank scales (voxel dependent)
            sigma: [B,C,D,H,W]   diagonal std dev
            B:     [C,r]         learned basis matrix (shared across voxels/batch)
            num_samples: int     number of Monte Carlo samples
            targets: [B,D,H,W]   (optional) target labels, unused here but kept for API compatibility

        Returns:
            logits_mc: [K,B,C,D,H,W]   sampled logits
        """
        Bsz, C, D, H, W = self.mu.shape
        r = self.low_rank_cov.shape[1]  # rank

        # -------------------------------------------------------
        # 1. Expand mean for sampling
        # -------------------------------------------------------
        mu_exp = self.mu.unsqueeze(0)  # [K,B,C,D,H,W]

        # -------------------------------------------------------
        # 2. Low-rank noise component
        # -------------------------------------------------------
        # eps_r ~ N(0, I), shape [K,B,r,D,H,W]
        eps_r = torch.randn(1, Bsz, r, D, H, W, device=self.mu.device, dtype=self.mu.dtype)

        # Scale by 'a': [K,B,r,D,H,W]
        scaled = eps_r * self.low_rank_cov.unsqueeze(0)

        # Reshape to [K,B,r,N] with N = D*H*W
        N = D * H * W
        scaled = scaled.view(1, Bsz, r, N)

        # Project into C with B:
        #   B: [C,r], scaled: [K,B,r,N]
        #   result: [K,B,C,N]
        low_rank = torch.matmul(self.cov_basis_mat, scaled)  

        # Reshape back to [K,B,C,D,H,W]
        low_rank_noise = low_rank.view(1, Bsz, C, D, H, W)

        # -------------------------------------------------------
        # 3. Diagonal noise component
        # -------------------------------------------------------
        eps_diag = torch.randn(1, Bsz, C, D, H, W, device=self.mu.device, dtype=self.mu.dtype)
        diag_noise = eps_diag * self.sigma.unsqueeze(0)

        # -------------------------------------------------------
        # 4. Combine all parts
        # -------------------------------------------------------
        logits_mc = mu_exp + low_rank_noise + diag_noise
        return logits_mc.squeeze(0)
    

class SamplerWeighted(BaseSampler):

    def __init__(self, mu, sigma, low_rank_cov, cov_basis_mat, weighting):
        super().__init__(mu, sigma, low_rank_cov, cov_basis_mat, weighting)
    

    def forward(self, ):
        """
        Sample MC logits from a Gaussian with low-rank + diagonal covariance.

            Distribution:
            logits ~ N(mu, Σ), with
            Σ = (B diag(a))(B diag(a))^T + diag(sigma^2)

        Args:
            mu:    [B,C,D,H,W]   mean logits
            a:     [B,r,D,H,W]   low-rank scales (voxel dependent)
            sigma: [B,C,D,H,W]   diagonal std dev
            B:     [C,r]         learned basis matrix (shared across voxels/batch)
            num_samples: int     number of Monte Carlo samples
            targets: [B,D,H,W]   (optional) target labels, unused here but kept for API compatibility

        Returns:
            logits_mc: [K,B,C,D,H,W]   sampled logits
        """

        num_samples = 1
        Bsz, C, D, H, W = self.mu.shape
        r = self.low_rank_cov.shape[1]  # rank

        weighting = self.weighting.unsqueeze(-1).unsqueeze(-1)

        cov_bases_selected = (weighting * self.cov_basis_mat.unsqueeze(0)).sum(dim = 1) # Here we sum over the different bases, for hard softmax only one value is none zero so silly to sum

        # -------------------------------------------------------
        # 1. Expand mean for sampling
        # -------------------------------------------------------
        mu_exp = self.mu.unsqueeze(0).expand(num_samples, -1, -1, -1, -1, -1)  # [K,B,C,D,H,W]

        # -------------------------------------------------------
        # 2. Low-rank noise component
        # -------------------------------------------------------
        # eps_r ~ N(0, I), shape [K,B,r,D,H,W]
        eps_r = torch.randn(num_samples, Bsz, r, D, H, W, device=self.mu.device, dtype=self.mu.dtype)

        # Scale by 'a': [K,B,r,D,H,W]
        scaled = eps_r * self.low_rank_cov.unsqueeze(0)

        # Reshape to [K,B,r,N] with N = D*H*W
        N = D * H * W
        scaled = scaled.view(num_samples, Bsz, r, N)

        # Project into C with B:
        #   B: [C,r], scaled: [K,B,r,N]
        #   result: [K,B,C,N]
        low_rank = torch.matmul(cov_bases_selected, scaled)  

        # Reshape back to [K,B,C,D,H,W]
        low_rank_noise = low_rank.view(num_samples, Bsz, C, D, H, W)

        # -------------------------------------------------------
        # 3. Diagonal noise component
        # -------------------------------------------------------
        eps_diag = torch.randn(num_samples, Bsz, C, D, H, W, device=self.mu.device, dtype=self.mu.dtype)
        diag_noise = eps_diag * self.sigma.unsqueeze(0)

        # -------------------------------------------------------
        # 4. Combine all parts
        # -------------------------------------------------------
        logits_mc = mu_exp + low_rank_noise + diag_noise
        return logits_mc
    



class ConvexSamplerWeighted(BaseSampler):

    def __init__(self, mu, sigma, low_rank_cov, cov_basis_mat, weighting):
        super().__init__(mu, sigma, low_rank_cov, cov_basis_mat, weighting)
    
        self.eps = 1e-8 
        self.per_basis_a = False

    def forward(self, ):
        """
        Sample Monte-Carlo logits from
        mu: [B, C, D, H, W]
        log_var: [B, C, D, H, W]  (diagonal log-variance)
        cov_bases: [N, C, r]       (N basis matrices)
        weights: either [B, N] (per-volume) or [B, N, D, H, W] (voxel-wise)
        a: either None, or
            - if per_basis_a=True: [B, N, r, D, H, W]  (a_n(x) per basis)
            - otherwise: [B, r, D, H, W]              (shared across bases)
        num_samples: int, number of MC samples to produce
        Returns:
        samples: tensor of shape [num_samples, B, C, D, H, W]
        """
        device = self.mu.device
        batch, C, D, H, W = self.mu.shape
        N, _, r = self.cov_basis_mat.shape  # cov_bases: [N, C, r]
     
        # Prepare weights -> make them voxel-wise [B, N, D, H, W]
        if self.weighting.dim() == 2:  # [B, N]
            w = self.weighting.view(batch, N, 1, 1, 1).expand(-1, -1, D, H, W)
        else:
            # assume already [B, N, D, H, W]
            w = self.weighting
        # small floor to avoid zero sqrt
        w = w.clamp(min=0.0)
        w_sqrt = torch.sqrt(w + self.eps)  # [B, N, D, H, W]

        # Pre-expand cov_bases for broadcast: [1, N, C, r, 1, 1, 1]
        cov_bases_exp = self.cov_basis_mat.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # diag noise term
        eps_diag = torch.randn_like(self.mu, device=device)  # [B,C,D,H,W]
        diag_term = self.sigma * eps_diag

        # ----- low-rank mixture term -----
        # Sample independent z_n for each basis
        # eps_low: [B, N, r, D, H, W]
        eps_low = torch.randn(batch, N, r, D, H, W, device=device)

        
        if self.per_basis_a:
            # a is [B, N, r, D, H, W]; multiply elementwise
            assert self.low_rank_cov.shape == (batch, N, r, D, H, W)
            scaled = self.low_rank_cov * eps_low
        else:
            # a is [B, r, D, H, W] -> broadcast to [B, N, r, D, H, W]
            assert self.low_rank_cov.shape == (batch, r, D, H, W)
            scaled = eps_low * self.low_rank_cov.unsqueeze(1)  # broadcast in N

        # Now compute B_n @ (a_n * z_n) for every n, batch, voxel.
        # Do this via broadcasting (no einsum):
        # - cov_bases_exp: [1, N, C, r, 1,1,1]
        # - scaled_unsq  : [B, N, 1, r, D,H,W]
        scaled_unsq = scaled.unsqueeze(2)  # [B, N, 1, r, D, H, W]
        # multiply -> [B, N, C, r, D, H, W]
        mult = (cov_bases_exp * scaled_unsq)  # broadcasting
        # sum over r -> [B, N, C, D, H, W]
        per_basis_lowrank = mult.sum(dim=3)

        # weight each basis contribution by sqrt(p_n(x)) and sum over N:
        # w_sqrt: [B, N, D, H, W] -> unsqueeze class dim to multiply
        w_sqrt_unsq = w_sqrt.unsqueeze(2)  # [B, N, 1, D, H, W]
        # multiply and sum over N -> [B, C, D, H, W]
        low_rank_sum = (w_sqrt_unsq * per_basis_lowrank).sum(dim=1)

        # final sample
        sample = self.mu + diag_term + low_rank_sum  # [B, C, D, H, W]
        return sample


class PartitionedBasisSampler(nn.Module):
    def __init__(self,mu, sigma, low_rank_cov, cov_basis_mat, weighting, num_parts=4):
        super().__init__()
        self.C = cov_basis_mat.shape[1]
        self.rank = cov_basis_mat.shape[-1]
        self.N = cov_basis_mat.shape[0]
        self.num_parts = num_parts

        self.B_large = cov_basis_mat
        self.probs = weighting
        self.mu = mu
        self.A = low_rank_cov
        self.sigma = sigma

    
    def sample(self, ):
        return self.forward()

    def partition_tensor(self, A, num_parts=4):
        """
        A: [B, Rank, H, W, D]
        Returns: [B, num_parts**3, Rank, H//num_parts, W//num_parts, D//num_parts]
        """
        B, Rank, H, W, D = A.shape
        assert H % num_parts == 0 and W % num_parts == 0 and D % num_parts == 0, \
            "Volume dims must be divisible by num_parts"

        h, w, d = H // num_parts, W //num_parts, D // num_parts

        # Reshape into (grid × local size)
        A_parts = A.reshape(B, Rank,
                            num_parts, h,
                            num_parts, w,
                            num_parts, d)

        # Move grid dims up front: (B, num_parts, num_parts, num_parts, Rank, h, w, d)
        A_parts = A_parts.permute(0, 2, 4, 6, 1, 3, 5, 7)

        # Merge partitions into a single index: (B, num_parts**3, Rank, h, w, d)
        A_parts = A_parts.reshape(B, num_parts**3, Rank, h, w, d)
        return A_parts
    
    def forward(self,):
        """
        Args:
            A:     [B, C, rank, H, W, D]   -- scaling coefficients
            feat:  [B, C, H, W, D]         -- features for basis selection
            sigma: [B, 1, 1, 1, 1, 1] or scalar noise std

        Returns:
            samples: [B, C, H, W, D]       -- one MC sample per input
        """

        B, C, H, W, D = self.mu.shape
        
        G = self.num_parts**3
        probs = self.probs.permute(0, 2, 3, 4, 1).reshape(B, G, self.N)   # [B, G, N]
        B_sel = torch.einsum("bgn,ncr->bgcr", probs, self.B_large)   # [B, G, C, rank]
        # ---- 2) Partition A into G blocks ----

        A_parts = self.partition_tensor(self.A)
        eps_rank = torch.randn_like(A_parts)  # [B, G, Rank, h, w, d]
        # Project with A
        proj = A_parts * eps_rank  # elementwise multiply rank dim
        # Sum over rank with B_sel
        lowrank_term = torch.einsum("bgcr,bgrhwd->bgchwd", B_sel, proj)

        # ---- 4) Reshape back into full volume ----
        lowrank_term = lowrank_term.view(B,
                            self.num_parts, self.num_parts, self.num_parts,
                            C,
                            H // self.num_parts, W // self.num_parts, D // self.num_parts)

        lowrank_term = lowrank_term.permute(0, 4, 1, 5, 2, 6, 3, 7) \
                        .reshape(B, C, H, W, D)

        # ---- 3) Add isotropic diagonal noise ----
        eps_iso = torch.randn_like(lowrank_term)
        samples = self.mu + lowrank_term + self.sigma * eps_iso
    
        return samples
    

class ProperPartionedSampler(PartitionedBasisSampler):

    def __init__(self, mu, sigma, low_rank_cov, cov_basis_mat, weighting, num_parts=4):
        super().__init__(mu, sigma, low_rank_cov, cov_basis_mat, weighting, num_parts)

    
    def forward(self, ):
        
        B, C, H, W, D = self.mu.shape
        
        G = self.num_parts**3
        probs = self.probs.permute(0, 2, 3, 4, 1).reshape(B, G, self.N)   # [B, G, N]
        B_sel = torch.einsum("bgn,ncr->bgcr", probs, self.B_large)   # [B, G, C, rank]

        """
        A_parts = (
            self.A.unfold(-3, H // self.num_parts, H // self.num_parts)
            .unfold(-2, W // self.num_parts, W // self.num_parts)
            .unfold(-1, D // self.num_parts, D // self.num_parts)
        )
        # Reorder dims -> [B, np, np, np, C, rank, h’, w’, d’]
        A_parts = A_parts.permute(0, 3, 5, 7, 1, 2, 4, 6, 8)

        # Flatten partitions -> [B, G, C, rank, h’, w’, d’]
        A_parts = A_parts.reshape(
            B, self.num_parts**3, C, self.rank,
            H // self.num_parts, W // self.num_parts, D // self.num_parts
        )
        """
        # ---- 2) Partition A into G blocks ----

        A_parts = self.partition_tensor(self.A)
        #eps_rank = torch.randn_like(A_parts)  # [B, G, Rank, h, w, d]
        eps_rank = torch.rand(B, A_parts.shape[1], 1, 1, 1,1, dtype = self.mu.dtype, device = self.mu.device)

        # Project with A
        proj = A_parts * eps_rank  # elementwise multiply rank dim
        # Sum over rank with B_sel
        lowrank_term = torch.einsum("bgcr,bgrhwd->bgchwd", B_sel, proj)

        # ---- 4) Reshape back into full volume ----
        lowrank_term = lowrank_term.view(B,
                            self.num_parts, self.num_parts, self.num_parts,
                            C,
                            H // self.num_parts, W // self.num_parts, D // self.num_parts)

        lowrank_term = lowrank_term.permute(0, 4, 1, 5, 2, 6, 3, 7) \
                        .reshape(B, C, H, W, D)

        # ---- 3) Add isotropic diagonal noise ----
        eps_iso = torch.randn_like(lowrank_term)
        samples = self.mu + lowrank_term + self.sigma * eps_iso
    
        return samples




class PerVoxelBasisSampler(nn.Module):
    """
    Sample from N voxel-specific low-rank covariances built from
    per-voxel weighted combinations of basis matrices.
    """
    def __init__(self, mu, sigma, low_rank_cov, cov_basis_mat, weighting):
        super().__init__()
        self.mu = mu                      # [B, C, H, W, D]
        self.sigma = sigma                # scalar or [B, 1, H, W, D]
        self.A = low_rank_cov             # [B, R, H, W, D]
        self.B_bases = cov_basis_mat      # [N, C, R]
        self.weighting = weighting        # [B, N, H, W, D]
        
        self.N_bases = cov_basis_mat.shape[0]
        self.C = cov_basis_mat.shape[1]
        self.rank = cov_basis_mat.shape[2]

    def forward(self):
        B, C, H, W, D = self.mu.shape
        N, _, R = self.B_bases.shape

        # 1. Combine basis matrices per voxel
        # weighting: [B, N, H, W, D]
        # B_bases: [N, C, R]
        # result: [B, C, R, H, W, D]
        B_voxel = torch.einsum('bnxyz,ncr->bcrxyz', self.weighting, self.B_bases)

        # 2. Sample rank latent variable per voxel
        eps_rank = torch.randn_like(self.A)       # [B, R, H, W, D]

        # 3. Project through A (elementwise multiply)
        proj = self.A * eps_rank                  # [B, R, H, W, D]

        # 4. Contract rank dimension through B_voxel
        # B_voxel: [B, C, R, H, W, D]
        # proj:    [B, R, H, W, D]
        # → [B, C, H, W, D]
        lowrank_term = torch.einsum('bcrxyz,brxyz->bcxyz', B_voxel, proj)

        # 5. Add diagonal Gaussian noise
        eps_iso = torch.randn_like(lowrank_term)
        samples = self.mu + lowrank_term + self.sigma * eps_iso

        return samples

    def sample(self):
        return self.forward()