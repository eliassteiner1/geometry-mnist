import numpy as np
import einops as eo
from scipy.interpolate import make_splprep


DIM = 3


def _sig(x: np.ndarray) -> np.ndarray:
    """ basically a sigmoid to get a smooth transition from y=0 to y=1 over 1x (unitsquare) """
    a = 12
    return 1 / (1 + np.exp(-a * (x - 0.5)))

def _hop(x: np.ndarray) -> np.ndarray:
    """ a gaussian bell to hop in z direction for the active crossing braid (unitsquare) """
    a = 30
    b = 0.4
    return b * np.exp(-a * (x - 0.5)**2)

def _generate_straight(n_pts: int) -> np.ndarray:
    """ generate just the straight line braid within unit square as a chain of points"""
    
    pts = np.zeros(shape=(n_pts, DIM))
    pts[:, 0] = np.linspace(0, 1, n_pts, endpoint=False)
    return pts
    
def _generate_l2r_active(n_pts: int) -> np.ndarray:
    """ the L->R transition with a positive zhop. hop can easily be made to go over or under by sign flip in z. this element is all exactly in the unitsquare """
    
    pts = np.zeros(shape=(n_pts, DIM))
    pts[:, 0] = np.linspace(0, 1, n_pts, endpoint=False)
    pts[:, 1] = _sig(pts[:, 0])
    pts[:, 2] = _hop(pts[:, 0])
    return pts
    
def _generate_r2l_passive(n_pts: int) -> np.ndarray:
    """ the R->L transition without zhop. this element is all exactly in the unitsquare """
    
    pts = np.zeros(shape=(n_pts, DIM))
    pts[:, 0] = np.linspace(0, 1, n_pts, endpoint=False)
    pts[:, 1] = 1 - _sig(pts[:, 0])
    return pts
    
def _generate_newslice(n_pts: int, n_braids: int) -> np.ndarray:
    """ helper to generate one op-slice for all braids and fill it with dummy straight lines for each braid. """
    
    # the base line element for one braid at y=z=0 with an extra batch braid dimension for broadcasting
    one_line = _generate_straight(n_pts=n_pts)[None, :, :]
    
    # a broadcasting matrix to duplicate the basis line n_braid times and increase y coords by 1 each time
    distributor = np.ones(shape=(n_braids, 1, DIM))
    distributor[:, 0, 1] = np.arange(n_braids) + 1
    
    # elementwise multiplication and broadcasting duplicates the line element with an increasing y coord. temporarily setting the initial y to 1 allows the multiplication to take effect
    one_line[:, :, 1] = 1
    bpts = one_line * distributor
    bpts[:, :, 1] -= 1 # set the line elements to begin at y=0 again

    return bpts

def _generate_braid_blocks(word: list, n_braids: int, n_ops: int, n_pts_per_ops: int) -> np.ndarray:
    
    # generate the empty per-braid-pointlist
    bpts = np.zeros(shape=(n_braids, n_ops*n_pts_per_ops, 3))

    # build word as points
    for i, op in enumerate(word):
        
        # extract braid_ids and hop_dir; braid indices here must be 0 indexed (the word is 1 indexed)!
        braid_id_active  = (abs(op) - 1)     # index of the braid that is targeted by the op
        braid_id_passive = (abs(op) - 1) + 1 # index of the braid that has to move as well to swap with the active one
        hop_dir = np.sign(op) # whether to hop over or under

        # one fresh "op slice"; all lists just get a straight line
        bpts_newslice = _generate_newslice(n_pts=n_pts_per_ops, n_braids=n_braids)
        
        # then overwrite the two braids that are part of the op with crossing points!
        pts_active = _generate_l2r_active(n_pts=n_pts_per_ops) # the active braid going from left to right
        pts_passive = _generate_r2l_passive(n_pts=n_pts_per_ops) # the passive braid (next) going from right to left
        # adjust y coordinates of points to braid position (they are all generated in the unit square)
        pts_active[:, 1]  += braid_id_active
        pts_passive[:, 1] += braid_id_active
        # finally flip sign of z-coords depending on hop direction (to go under or over the passive braid)
        pts_active[:, 2] *= hop_dir 
        # overwrite the straight lines with the two crossings
        bpts_newslice[braid_id_active, :, :] = pts_active
        bpts_newslice[braid_id_passive, :, :] = pts_passive

        # offset the entire new slice to the current "x end position"
        bpts_newslice[:, :, 0] += i
        
        # write the newslice into the main braid-points array
        bpts[:, i*n_pts_per_ops:(i+1)*n_pts_per_ops, :] = bpts_newslice

    # now convert these to proper xyz coords (before it was generated basically in phi-y-r). a certain minimum radius makes sure that the negative hops don't cross through the middle for very short words. the y axis points alogn the braid center axis.
    R = max(1, n_ops / (2*np.pi))
    bpts_xyz = np.zeros_like(bpts)
    bpts_xyz[:, :, 0] = (R + bpts[:, :, 2]) * np.sin(bpts[:, :, 0]/n_ops * 2*np.pi)
    bpts_xyz[:, :, 1] = bpts[:, :, 1]
    bpts_xyz[:, : ,2] = (R + bpts[:, :, 2]) * np.cos(bpts[:, :, 0]/n_ops * 2*np.pi)
    
    return bpts_xyz

def _link_braid_blocks(bpts: np.ndarray, n_braids: int, n_ops: int, n_pts_per_ops: int) -> np.ndarray:
    
    # all the points (after the blocks have been linked together) will be one single long contiguous list
    pts = np.zeros(shape=(n_ops*n_braids*n_pts_per_ops, 3))

    # decompose the per-braid lists into it's individual operations and create one list of all the individual op. these sequences are known to be connected points within itself - one "block".
    block_list = eo.rearrange(bpts, "braid (block pts) xyz -> (braid block) pts xyz", pts=n_pts_per_ops)

    # ----- for each LAST point, find the corresponding closest next FISRT point -----

    # extract only the beginning and end points of each block "chain"
    first_points = block_list[:, 0, :] # [n_first_pts, 3]
    last_points  = block_list[:, -1, :] # [n_last_pts, 3]

    # batched point differences
    diffs = (
        eo.rearrange(last_points, "n_last xyz -> n_last 1 xyz") -
        eo.rearrange(first_points, "n_first xyz -> 1 n_first xyz") 
    )
    squared_dist = eo.reduce(diffs**2, "n_last n_first xyz -> n_last n_first", "sum")

    # argmin finds all the links, but they are unordered!
    successor_map = np.argmin(squared_dist, axis=1)
    
    # ----- link the blocks -----
    
    # link all the blocks into one list (list of their indices from the block_list)
    if len(set(successor_map)) != len(successor_map):
        raise ValueError("ambiguous linking: some block has multiple predecessors")

    block_id = 0
    order = []
    visited = set()

    for _ in range(len(successor_map)):
        # if for some reason, a block is visited twice without having taken n_blocks steps, a loop is distjoint
        if block_id in visited:
            raise ValueError(f"knot is made up of disjoint loops!")
        visited.add(int(block_id))
        order.append(block_id)
        block_id = successor_map[block_id]
    
    # grab all the block in the right order and rearrange them into one long list
    pts = eo.rearrange(block_list[order], "allblocks pts xyz -> (allblocks pts) xyz")

    return pts

def generate_braid_closure(word: list, n_pts_per_ops: int = 30) -> np.ndarray:
    
    # determined params
    n_braids = np.max(np.abs(word)) + 1
    n_ops = len(word) 

    # create braid as one long list of connected points (circulant)
    bpts = _generate_braid_blocks(word, n_braids, n_ops, n_pts_per_ops)
    pts  = _link_braid_blocks(bpts, n_braids, n_ops, n_pts_per_ops)

    return pts

def resample_normalize_chain(pts: np.ndarray, n_resample: int, smoothing: float = 0.0) -> np.ndarray:
    """ resamples closed chain at regular arc-intervals to desired number of points and normalizes total ropelength """
    
    # fitting a spline through all rough points and resampling at uniform arc-intervals
    spl, _ = make_splprep(x=np.vstack([pts, pts[:1]]).T, k=3, s=smoothing, bc_type="periodic")
    pts_dense = spl(np.linspace(0, 1, n_resample, endpoint=False)).T # resampling with arbitrary number of points

    # normalize the ropelength to 1
    dists  = np.linalg.norm(pts_dense - np.roll(pts_dense, shift=1, axis=0), axis=1)
    length = np.sum(dists)
    pts_dense = pts_dense - pts_dense.mean(0)
    pts_dense = pts_dense / length
    
    return pts_dense

