import torch
from dataclasses import dataclass

from geometry_mnist.knotgen.chain import ChainProps
from geometry_mnist.knotgen.chain import calc_distances_squared

# some constants that were determined by running all energies on a ring of unit circumference and 1024 points. this should help to pre-balance all the contributions somewhat. # TODO: persistent notebook that shows how to derive!
NCONST_CONF = (1/00.0127)
NCONST_BEND = (1/19.7400)
NCONST_FREP = (1/99.7000)

@dataclass(frozen=True)
class EnergyWeights: 
    conf: float
    bend: float
    frep: float


def _e_confinement(pts: torch.Tensor) -> torch.Tensor:
    """ the further away from the origin all points are the more energy. use square of distances to not have a linear energy and no kink at origin. basically a bowl shaped potential """
    
    distances_from_origin_squared = (pts**2).sum(dim=1)
    e = distances_from_origin_squared.sum()
    return e

def _e_bending(pts: torch.Tensor) -> torch.Tensor:
    """ forces two consecutive segments to be colinear. going via the angle is unnecessary as an arccos corrupts the gradients. instead 1 - cos(phi) is used directly as the "bending energy". this is also square weighted as for small angles. the 1 - cos(phi) is ~= phi^2 / 2 """
    
    v_prev = pts - pts.roll(shifts=1, dims=0)  # x_i − x_{i−1}: edge from i−1 to i
    v_next = pts.roll(shifts=-1, dims=0) - pts # x_{i+1} − x_i: edge from i to i+1
    cos_phi = torch.einsum("ij, ij -> i", v_prev, v_next) / (v_prev.norm(dim=1) * v_next.norm(dim=1))
    e = (1 - cos_phi).sum()
    return e
    
def _e_repulse_far(pts_dst: torch.Tensor, far_pairs_mask: torch.Tensor) -> torch.Tensor:
    """ far reaching point repulsion, to separate out entagnled strands, mostly for relaxation; all points repel each other pairwise by 1/dist^2 """
    
    # energy is just 1/dist^2 but we mask off the degenerate ii pairs and the ones close on the arc. dont count repulsion between points that are quite close along the arc. clip the diagonal 0 distance values so that division works, result will always be masked anyways.
    e = (far_pairs_mask * (1 / pts_dst.clip(min=1e-12, max=None))).sum()
    return e

def e_total(
    pts: torch.Tensor, pts_dst: torch.Tensor, 
    chain_props: ChainProps,
    e_weights: EnergyWeights,
) -> torch.Tensor:
    """ individual contributions have be normalized to a circle of unit circumference """

    e = (0
        + e_weights.conf * NCONST_CONF * _e_confinement(pts) / 2 * chain_props.linklen
        + e_weights.bend * NCONST_BEND * _e_bending(pts) / chain_props.linklen
        + e_weights.frep * NCONST_FREP * _e_repulse_far(pts_dst, chain_props.far_pairs_mask) * chain_props.linklen**2
    )
    return e

def e_total_with_grad(
    pts: torch.Tensor,
    chain_props: ChainProps,
    e_weights: EnergyWeights,   
) -> tuple[float, torch.Tensor, torch.Tensor]:  
    
    # autograd operations and distance matrix can be reused later
    pts_autograd = pts.detach().requires_grad_(True) # only this local copy is ever tracked
    pts_dst_autograd = calc_distances_squared(pts_autograd)
    
    e = e_total(
        pts=pts_autograd, pts_dst=pts_dst_autograd,
        chain_props=chain_props,
        e_weights=e_weights,
    )
    (g, ) = torch.autograd.grad(e, pts_autograd)
    
    return e.item(), g, pts_dst_autograd.detach()
    






