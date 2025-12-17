"""
Uncertainty computation for USplat4D (Equation 3, 5, 6).

This module computes per-Gaussian uncertainty from rendering statistics
and transforms it into anisotropic 3D covariance matrices.
"""

import torch
from typing import Tuple, Optional


def compute_gaussian_uncertainty(
    gaussians_per_pixel: torch.Tensor,  # [H, W, max_gaussians] - indices
    transmittance_alpha: torch.Tensor,  # [H, W, max_gaussians] - T * alpha
    num_gaussians: int,
    min_variance_inv: float = 1e-6,
    convergence_threshold: Optional[float] = None,
    rendered_image: Optional[torch.Tensor] = None,
    target_image: Optional[torch.Tensor] = None,
    phi_constant: float = 1e3,
) -> torch.Tensor:
    """
    Compute per-Gaussian uncertainty according to Equation 3 and 5.
    
    Equation 3: σ²ᵢ,ₜ = [Σₕ∈Ωᵢ,ₜ (Tʰᵢ,ₜ αᵢ)²]⁻¹
    
    Where:
    - Ωᵢ,ₜ: set of pixels to which Gaussian i contributes at time t
    - Tʰᵢ,ₜ: transmittance of Gaussian i at pixel h
    - αᵢ: opacity of Gaussian i
    
    Equation 5 (optional convergence check):
    uᵢ,ₜ = Iᵢ,ₜ · σ²ᵢ,ₜ + (1 - Iᵢ,ₜ) · φ
    
    Where:
    - Iᵢ,ₜ = min(I(h)) over h ∈ Ωᵢ,ₜ
    - I(h) = 1 if ||C̄ʰ - Cʰ||₁ < ηc, else 0
    - φ: large constant for non-converged regions
    
    Args:
        gaussians_per_pixel: [H, W, max_gaussians] indices of contributing Gaussians
        transmittance_alpha: [H, W, max_gaussians] blending weights (T * alpha)
        num_gaussians: Total number of Gaussians
        min_variance_inv: Minimum variance inverse to prevent division by zero
        convergence_threshold: If provided, use Eq. 5 with this as ηc
        rendered_image: [H, W, 3] for convergence check (Eq. 5)
        target_image: [H, W, 3] for convergence check (Eq. 5)
        phi_constant: Large constant φ for non-converged pixels (Eq. 5)
    
    Returns:
        uncertainty: [num_gaussians] scalar uncertainty per Gaussian
    """
    H, W, max_gaussians = gaussians_per_pixel.shape
    device = gaussians_per_pixel.device
    
    # Accumulate variance inverse: Σ (T * alpha)²
    variance_inv = torch.zeros(num_gaussians, device=device, dtype=torch.float32)
    
    # Flatten spatial dimensions for easier processing
    pixel_indices = gaussians_per_pixel.reshape(-1, max_gaussians)  # [H*W, max_gaussians]
    weights = transmittance_alpha.reshape(-1, max_gaussians)  # [H*W, max_gaussians]
    
    # Create valid mask (gaussian_idx >= 0)
    valid_mask = pixel_indices >= 0  # [H*W, max_gaussians]
    
    # Get valid indices and weights
    valid_indices = pixel_indices[valid_mask]  # [N_valid]
    valid_weights = weights[valid_mask]  # [N_valid]
    
    # Accumulate squared weights per Gaussian (Eq. 3)
    variance_inv.index_add_(0, valid_indices, valid_weights ** 2)
    
    # Clamp to prevent division by zero
    variance_inv = torch.clamp(variance_inv, min=min_variance_inv)
    
    # Compute scalar uncertainty
    uncertainty = 1.0 / variance_inv  # σ²ᵢ,ₜ
    
    # Optional: Apply convergence indicator (Eq. 5)
    if convergence_threshold is not None and rendered_image is not None and target_image is not None:
        # Compute per-pixel L1 error
        pixel_error = torch.abs(rendered_image - target_image).sum(dim=-1)  # [H, W]
        pixel_converged = (pixel_error < convergence_threshold).float()  # [H, W]
        
        # For each Gaussian, find minimum convergence indicator across its pixels
        gaussian_converged = torch.ones(num_gaussians, device=device, dtype=torch.float32)
        
        pixel_converged_flat = pixel_converged.reshape(-1)  # [H*W]
        for i in range(max_gaussians):
            mask = valid_mask[:, i]  # [H*W]
            indices = pixel_indices[:, i][mask]  # Contributing Gaussians for this layer
            converged = pixel_converged_flat[mask]  # Convergence status
            
            # Update: take minimum (if any pixel is non-converged, Gaussian is non-converged)
            gaussian_converged.index_reduce_(
                0, indices, converged, reduce='amin', include_self=True
            )
        
        # Apply Eq. 5: u = I * σ² + (1 - I) * φ
        uncertainty = gaussian_converged * uncertainty + (1 - gaussian_converged) * phi_constant
    
    return uncertainty


def transform_to_anisotropic(
    uncertainty: torch.Tensor,  # [N] scalar uncertainty
    positions: torch.Tensor,  # [N, 3] Gaussian positions
    camera_rotation: torch.Tensor,  # [3, 3] camera-to-world rotation
    is_monocular: bool,
    rx: float = 1.0,
    ry: float = 1.0,
    rz_monocular: float = 5.0,
    rz_multiview: float = 1.0,
) -> torch.Tensor:
    """
    Transform scalar uncertainty into anisotropic 3D covariance (Equation 6).
    
    Equation 6: Uᵢ,ₜ = Rₓc · diag(rₓ uᵢ,ₜ, rᵧ uᵢ,ₜ, rᵤ uᵢ,ₜ) · Rₓcᵀ
    
    Where:
    - Rₓc: camera-to-world rotation matrix (Rwc in paper notation)
    - (rₓ, rᵧ, rᵤ): axis-aligned scaling factors
    - uᵢ,ₜ: scalar uncertainty from Eq. 3 or 5
    
    Args:
        uncertainty: [N] scalar uncertainty values
        positions: [N, 3] Gaussian positions (unused here, for future extensions)
        camera_rotation: [3, 3] camera-to-world rotation matrix
        is_monocular: If True, use larger rz for depth ambiguity
        rx: Scaling factor for x-axis (lateral)
        ry: Scaling factor for y-axis (lateral)
        rz_monocular: Scaling factor for z-axis (depth) in monocular case
        rz_multiview: Scaling factor for z-axis in multi-view case
    
    Returns:
        covariance_matrices: [N, 3, 3] anisotropic uncertainty matrices
    """
    N = uncertainty.shape[0]
    device = uncertainty.device
    
    # Choose rz based on setup
    rz = rz_monocular if is_monocular else rz_multiview
    
    # Create diagonal uncertainty in camera space: diag(rx*u, ry*u, rz*u)
    Uc = torch.zeros(N, 3, 3, device=device, dtype=torch.float32)
    Uc[:, 0, 0] = rx * uncertainty
    Uc[:, 1, 1] = ry * uncertainty
    Uc[:, 2, 2] = rz * uncertainty
    
    # Transform to world space: U = R * Uc * R^T
    R = camera_rotation.unsqueeze(0)  # [1, 3, 3]
    U = R @ Uc @ R.transpose(-2, -1)  # [N, 3, 3]
    
    return U


def compute_mahalanobis_distance(
    pos_i: torch.Tensor,  # [N, 3] or [3]
    pos_j: torch.Tensor,  # [M, 3] or [3]
    cov_inv_i: torch.Tensor,  # [N, 3, 3] or [3, 3]
    cov_inv_j: Optional[torch.Tensor] = None,  # [M, 3, 3] or [3, 3], if None use cov_inv_i
) -> torch.Tensor:
    """
    Compute Mahalanobis distance for UA-kNN (used in Equation 7).
    
    Distance: d² = (xᵢ - xⱼ)ᵀ Uᵢ⁻¹ (xᵢ - xⱼ)
    
    Args:
        pos_i: [N, 3] positions of query Gaussians
        pos_j: [M, 3] positions of candidate Gaussians
        cov_inv_i: [N, 3, 3] inverse covariance matrices for query Gaussians
        cov_inv_j: [M, 3, 3] optional inverse covariance for candidates (for symmetric distance)
    
    Returns:
        distances: [N, M] Mahalanobis distances squared
    """
    # Handle single query case
    if pos_i.dim() == 1:
        pos_i = pos_i.unsqueeze(0)  # [1, 3]
        cov_inv_i = cov_inv_i.unsqueeze(0)  # [1, 3, 3]
    
    if pos_j.dim() == 1:
        pos_j = pos_j.unsqueeze(0)  # [1, 3]
    
    N = pos_i.shape[0]
    M = pos_j.shape[0]
    
    # Compute pairwise differences: [N, M, 3]
    diff = pos_i.unsqueeze(1) - pos_j.unsqueeze(0)  # [N, M, 3]
    
    # Compute (x - y)^T U^{-1} (x - y) for each pair
    # diff: [N, M, 3], cov_inv_i: [N, 3, 3]
    # Expand cov_inv_i: [N, 1, 3, 3] to broadcast over M
    cov_inv_expanded = cov_inv_i.unsqueeze(1)  # [N, 1, 3, 3]
    
    # temp = U^{-1} @ (x - y): [N, M, 3]
    temp = torch.einsum('nicd,nmd->nmc', cov_inv_expanded, diff)  # [N, M, 3]
    
    # distances = (x - y)^T @ temp: [N, M]
    distances_sq = torch.einsum('nmd,nmd->nm', diff, temp)
    
    return distances_sq


def invert_covariance_safe(
    covariance: torch.Tensor,  # [N, 3, 3]
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Safely invert covariance matrices with regularization.
    
    Args:
        covariance: [N, 3, 3] covariance matrices
        epsilon: Regularization term added to diagonal
    
    Returns:
        covariance_inv: [N, 3, 3] inverted covariance matrices
    """
    N = covariance.shape[0]
    device = covariance.device
    
    # Add regularization: U + ε*I
    regularized = covariance + epsilon * torch.eye(3, device=device).unsqueeze(0)
    
    # Invert using torch.linalg.inv (more stable than manual inversion)
    covariance_inv = torch.linalg.inv(regularized)
    
    return covariance_inv
