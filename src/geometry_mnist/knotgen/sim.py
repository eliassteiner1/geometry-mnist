from dataclasses import dataclass, astuple
import torch

from geometry_mnist.knotgen.chain import ChainProps, calc_total_ropelength
from geometry_mnist.knotgen.energy import EnergyWeights, e_total_with_grad


@dataclass(frozen=True)
class SimParams:
    
    dt_limit: float
    temperature: float = 0.00
    max_retries: int = 5
    max_strand_gap_closure: float = 0.25
    max_link_strain: float = 0.10
    sigma_security: float = 4.00 # for what counts as the empirical "max" of noise distribution
    reproject_max_iters: int = 100
    reproject_rtol: float = 0.01

@dataclass    
class DtAdmissible: 
    pushabs: float
    pushdff: float
    kickabs: float
    kickdff: float
    
    @property
    def binding(self) -> float: 
        return min(astuple(self))
    
    @property
    def binding_reason(self) -> str:
        dct = vars(self)
        return min(dct, key=dct.get)

@dataclass
class StepResult:
    
    pts_next: torch.Tensor
    pts_eff_step: torch.Tensor
    g: torch.Tensor
    e: float
    dt_eff: float
    dt_eff_reason: str
    n_retries: int
    
    
  
def _reproject_links(pts: torch.Tensor, chain_props: ChainProps, sim_params: SimParams) -> torch.Tensor:
    
    """ PBD: pull each link back to length l0, even/odd edges alternating. n_points must be even! """
    
    n_pts = len(pts)
    p_idx = torch.arange(n_pts)
    
    for _ in range(sim_params.reproject_max_iters):
        
        for start in (0, 1):
            # get even or odd point indices, depending on start 0 or 1
            i     = p_idx[start::2]
            inext = (i + 1) % n_pts # modulo to wrap around to first point when shifting for start=1

            # apply half of the linklength error correction to each point to bring them closer or further
            linkvecs   = pts[inext] - pts[i]
            lengths    = linkvecs.norm(dim=1, keepdim=True)
            corrvecs   = 0.5 * (lengths - chain_props.linklen) * (linkvecs / lengths) # delt_length * unitvec = corrvec
            pts[i]     = pts[i] + corrvecs
            pts[inext] = pts[inext] - corrvecs
            
        # keep track of what the maximum relative linklenght error was for the full chain
        lengths = (pts - pts.roll(shifts=1, dims=0)).norm(dim=1)
        max_rel_err = ((lengths - chain_props.linklen).abs().max() / chain_props.linklen).item()

        # early stopping if all relative errors on the link lengths are below rtol
        if max_rel_err < sim_params.reproject_rtol:
            return pts
    
    # if returned after max_iters, not all linklengths have the desired rtol!
    return pts 
    
def _max_admissible_dt(
    g: torch.Tensor, pts_dst: torch.Tensor,
    chain_props: ChainProps,
    sim_params: SimParams,
) -> DtAdmissible:
    
    """ finds the maxium timestep that is possible with the current chain layout - so that no limits are violated: no strands can come too close to each other (max_strand_gap_closure * min_present_strand_sep) and no links should be stretched too much (max_link_strain * linklen). for Push its possible to find the actual maximum value and then set dt_admissible according to that. for Kick (noise) empirical maximum values are used. """
    
    min_strand_separation = torch.sqrt(pts_dst[chain_props.far_pairs_mask].min())
    displacement_limit_strandsep = sim_params.max_strand_gap_closure * min_strand_separation
    displacement_limit_links =sim_params.max_link_strain * chain_props.linklen
    
    # ===== limits for PUSH 
    push_per_dt = -g/chain_props.linklen
    max_push_per_dt = push_per_dt.norm(dim=1).max() # how much ist teh 
    max_pushdiff_per_dt = (push_per_dt - push_per_dt.roll(shifts=1, dims=0)).norm(dim=1).max() # from point to point
    
    dt_max_push = displacement_limit_strandsep / max_push_per_dt
    dt_max_pushdiff = displacement_limit_links / max_pushdiff_per_dt
    
    # ===== limits for KICK
    if sim_params.temperature > 0:
        # these only make sense if there is a temperature at all
        dt_max_empiric_kick = (1
            * displacement_limit_strandsep**2 
            * (chain_props.linklen / (2 * sim_params.temperature))
            * (1 / (sim_params.sigma_security**2 * chain_props.sobolev_k1_gain_point**2))
        )
        dt_max_empiric_kickdiff = (1
            * displacement_limit_links**2
            * (chain_props.linklen / (2 * sim_params.temperature))
            * (1 / (sim_params.sigma_security**2 * chain_props.sobolev_k1_gain_diff**2))
        )
    else: 
        # if no temperature, it's dt limits should be ignored under min
        dt_max_empiric_kick = float("inf")
        dt_max_empiric_kickdiff = float("inf")
        
    dt_admissible = DtAdmissible(
        pushabs=float(dt_max_push),
        pushdff=float(dt_max_pushdiff),
        kickabs=float(dt_max_empiric_kick),
        kickdff=float(dt_max_empiric_kickdiff),
    )
    
    return dt_admissible

def _verify_effective_step(
    pts: torch.Tensor, pts_dst: torch.Tensor, pts_next,
    chain_props: ChainProps,
    sim_params: SimParams,
):
    """ once an update step (push + kick) has been applied and the links have been projected back to a valid configuration, it's possible that limits have been violated (push + kick stacking up unfavourably or PBD reproject has to do too much work). check this violation after applying update """

    # find the applicable limits (same as for the admissibe dt)
    min_strand_separation = torch.sqrt(pts_dst[chain_props.far_pairs_mask].min())
    displacement_limit_strandsep = sim_params.max_strand_gap_closure * min_strand_separation
    displacement_limit_links = sim_params.max_link_strain * chain_props.linklen
    
    # this is now [Push + Kick + reprojection] -> find maxium values of this "overall change"
    eff_step = pts_next - pts 
    max_step = eff_step.norm(dim=1).max()
    max_stepdiff = (eff_step - eff_step.roll(shifts=1, dims=0)).norm(dim=1).max()
      
    if max_step > displacement_limit_strandsep:
        return False
    if max_stepdiff > displacement_limit_links:
        return False
    
    return True
   
def simstep(
    pts: torch.Tensor, # the main state
    chain_props: ChainProps, # static stuff regarding the chain
    e_weights: EnergyWeights, # energy balancing weights
    sim_params: SimParams, # temp, dt_limits, etc
):
    
    # ===== calculating the total energy term and differentiating it to find the gradients
    e, g, pts_dst = e_total_with_grad(pts, chain_props=chain_props, e_weights=e_weights)
    
    # preconditioning (smoothing) the gradients
    g = chain_props.sobolev_k2@g
    
    # ===== dt checks (find the effective largest dt that this step can run with)
    dt_admissible = _max_admissible_dt(g, pts_dst, chain_props, sim_params)
    if dt_admissible.binding < sim_params.dt_limit:
        dt_eff = dt_admissible.binding
        dt_eff_reason = dt_admissible.binding_reason
    else: 
        dt_eff = sim_params.dt_limit
        dt_eff_reason = "limiter"
    
    for r in range(sim_params.max_retries):
        
        # ===== actual update step - but with dt_eff
        l0 = chain_props.linklen
        T = sim_params.temperature
        dx = (-g/l0)*dt_eff + (2*T*dt_eff/l0)**0.5 * (chain_props.sobolev_k1@torch.randn_like(pts))
        pts_next = pts + dx
        
        # ===== apply constraints
        pts_next = pts_next - pts_next.mean(dim=0) # center
        pts_next = pts_next / calc_total_ropelength(pts_next)
        pts_next = _reproject_links(pts_next, chain_props, sim_params)

        # ===== verify and possibly retry with dt_eff/2...
        step_valid = _verify_effective_step(pts, pts_dst, pts_next, chain_props, sim_params)

        if step_valid:
            n_retries = r
            break
        else:
            dt_eff = dt_eff/2 # retry simulation step with a smaller dt
    
    # this only hits if the loop finished without early breaking (=succeeding)
    else:
        raise RuntimeError(f"simulation step failed after reaching max retries!")

    # ===== return outputs
    result = StepResult(
        pts_next=pts_next,
        pts_eff_step=pts_next - pts,
        g=g,
        e=e,
        dt_eff=dt_eff,
        dt_eff_reason=dt_eff_reason,
        n_retries=n_retries
    )

    return result



