import torch
from dataclasses import dataclass

# some constants that were determined by running all energies on a ring of unit circumference and 1024 points. this should help to pre-balance all the contributions somewhat. # TODO: persistent notebook that shows how to derive!
NCONST_CONF = (1/00.0127)
NCONST_BEND = (1/19.7400)
NCONST_FREP = (1/99.7000)

@dataclass(frozen=True)
class EnergyW: 
    conf: float
    bend: float
    frep: float
    trep: float
    
    

def e_confinement(pts: torch.Tensor) -> torch.Tensor:
    """ the further away from the origin all points are the more energy. use square of distances to not have a linear energy and no kink at origin. basically a bowl shaped potential """
    
    distances_from_origin_squared = (pts**2).sum(dim=1)
    e = distances_from_origin_squared.sum()
    return e

def e_bending(pts: torch.Tensor) -> torch.Tensor:
    """ forces two consecutive segments to be colinear. going via the angle is unnecessary as an arccos corrupts the gradients. instead 1 - cos(phi) is used directly as the "bending energy". this is also square weighted as for small angles. the 1 - cos(phi) is ~= phi^2 / 2 """
    
    v_prev = pts - pts.roll(shifts=1, dims=0)  # x_i − x_{i−1}: edge from i−1 to i
    v_next = pts.roll(shifts=-1, dims=0) - pts # x_{i+1} − x_i: edge from i to i+1
    cos_phi = torch.einsum("ij, ij -> i", v_prev, v_next) / (v_prev.norm(dim=1) * v_next.norm(dim=1))
    e = (1 - cos_phi).sum()
    return e
    
def e_repulse_far(pts: torch.Tensor, pts_dst: torch.Tensor, repulsion_mask: torch.Tensor) -> torch.Tensor:
    """ far reaching point repulsion, to separate out entagnled strands, mostly for relaxation; all points repel each other pairwise by 1/dist^2 """
    
    # energy is just 1/dist^2 but we mask off the degenerate ii pairs and the ones close on the arc. dont count repulsion between points that are quite close along the arc. clip the diagonal 0 distance values so that division works, result will always be masked anyways.
    e = (repulsion_mask * (1 / pts_dst.clip(min=1e-12, max=None))).sum()
    return e

def e_repulse_tube(pts: torch.Tensor)  -> torch.Tensor:
    # would be more for simulating actual tubes that cannot intersect and would be used for the dataset generation
    raise(NotImplementedError)

def e_total(
    pts: torch.Tensor, pts_dst: torch.Tensor, repulsion_mask: torch.Tensor,
    linklen: float, 
    w_conf: float, w_bend: float, w_frep: float
) -> torch.Tensor:
    """ individual contributions have be normalized to a circle of unit circumference """
    
    # normalizing terms to make "dimensionless" as well as more or less equal magnitude TODO: explanation for the normalization of the "units" (making dimensionless)
    e = (0
        + w_conf * NCONST_CONF * e_confinement(pts) / 2 * linklen
        + w_bend * NCONST_BEND * e_bending(pts) / linklen
        + w_frep * NCONST_FREP * e_repulse_far(pts, pts_dst, repulsion_mask) * linklen**2
    )
    return e



def e_with_grad(pts: torch.Tensor):
    # x = pts.detach().requires_grad_(True) # only this local copy is ever tracked
    # E = e_total(x, **kwargs for energy)
    # g, *_ = torch.autograd.grad(E, x)
    
    # return E.item(), g
    
    # this way, in the main step I don't have to worry about what gets tracked by autograd and what not
    
    raise(NotImplementedError)
    






