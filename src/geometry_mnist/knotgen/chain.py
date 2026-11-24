import math
import einops as eo
import torch

# one time at the beginning, only NUMBER of points relevant ==========================


# this is not juuust for repulsion, but is more of a definition-mask for what is meant as being close neighbours on the chain and what isn't
def construct_repulsion_mask(n_pts: int, excl_percentage: float = 0.02) -> torch.Tensor:
    """ mask to exclude the points that are close on the arc (thick diagonal matrix); works for uniform link length. determine what's close as a percentage of the total ropelength (excl_percentage) """

    idx = torch.arange(n_pts)
    arc_distances = (idx[:, None] - idx[None, :]).abs()
    arc_distances = torch.minimum(arc_distances, n_pts - arc_distances) # assumes a looped chain, can go either way
    mask = arc_distances > excl_percentage*n_pts
    return mask

def construct_sobolev_kernel(n_points: int, reach_percentage: float = 0.02) -> torch.Tensor:
    """ generates the smoothing kernel matrix from sobolev gradients (this is H1 metric). the reach percentage is the "strength" of the smoothing; as a percentage of the entire chain length, it says at what point the kernel decays to less than 1/e (37%) strength (only valid ~0.2, then the wraparound effects of a closed chain start to become important). Note: the reach on the H2 squared version for gradients is ~2 times! """
    
    # define alpha in terms of the entire chain length (more or less cutoff @37% strength of kernel)
    ell = reach_percentage * n_points
    if ell == 0:
        alpha = 0
    else:
        alpha = 1.0 / (4 * math.sinh(1 / (2*ell))**2)
    
    # Sobolev kernel (screened Poisson / Laplacian smoothing), periodic chain:
    #
    #   L = periodic 1D Laplacian:  (L u)_i = u_{i+1} - 2 u_i + u_{i-1}
    #   A = I - alpha * L
    #   K = A^{-1}                  ->  K_ij ~ exp(-|i-j| / ell)   (two-sided exponential smoothing)
    #
    #   alpha chosen so the decay length is exactly ell points:
    #       1/alpha = 2 cosh(1/ell) - 2 = 4 sinh^2(1 / (2 ell))      (alpha ~ ell^2 for large ell)
    #   ell = reach_percentage * n_points    -> K drops to 1/e (~37%) at distance ell
    #
    #   usage:  push uses K @ K   (H^2, matches bending; reach ~ 2.15 ell)
    #           noise uses K      (since K K^T = K^2  ->  correct temperature)
    I = torch.eye(n_points, dtype=torch.float64)
    L = -2*I  + I.roll(shifts=1, dims=0) + I.roll(shifts=-1, dims=0)
    A = I - alpha*L
    A_INV = torch.linalg.inv(A)

    return (A_INV).to(torch.float64)



# update whenever the points themselves have changed ================================

def calc_distances_squared(pts: torch.Tensor) -> torch.Tensor:
    """ square matrix of the distances between all point pairs """

     # transpose one copy of the points, broadcasting automatically happens for the difference
    dst = ((eo.rearrange(pts, "n dim -> n 1 dim") - eo.rearrange(pts, "n dim -> 1 n dim"))**2).sum(dim=2)
    return dst

def calc_total_ropelength(pts: torch.Tensor) -> torch.Tensor:
    """ find the total ropelength of a chain (assumed to be circulant) """
    
    dists  = (pts - pts.roll(shifts=1, dims=0)).norm(dim=1)
    length = dists.sum()
    return length