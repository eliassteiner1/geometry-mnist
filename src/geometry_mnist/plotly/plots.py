import plotly.graph_objects as go
import numpy as np

from geometry_mnist.plotly.templates import register
register()


def plot_bpts(bpts: np.ndarray, kos: bool = True, kos_len: float = 1.0) -> go.Figure:
    """ plots a list of point-lists (multiple braids for example or blocks)"""
    # add the main data
    data = [
        go.Scatter3d(
            x=block[:, 0],
            y=block[:, 1],
            z=block[:, 2],
            marker=dict(size=4),
        )
        for i, block in enumerate(bpts)
    ]
    
    # add a KOS
    if kos:
        data += [
            go.Scatter3d(
                x=[0, kos_len if ax=="x" else 0], y=[0, kos_len if ax=="y" else 0], z=[0, kos_len if ax=="z" else 0],
                mode="lines",
                line=dict(color=col, width=10),
                showlegend=False,
                hoverinfo="skip",
            )
            for ax, col in zip(["x", "y", "z"], ["red", "green", "blue"])
        ]
    
    # create figure
    fig = go.Figure(
        data=data,
        layout=dict(
            template="mine_dark+mine_3d",
            scene=dict(aspectmode="data"),
            height=700
        ),
    )
    return fig


def plot_pts(pts: np.ndarray, kos: bool = True, kos_len: float = 1.0) -> go.Figure:
    """ convenience for a 3d plot of a chain of points """
    
    # add the main data
    data = [
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            marker=dict(
                color=np.arange(len(pts)), 
                colorscale="rainbow", 
                size=4,
            ),
        )
    ]
    
    # add an optional KOS
    if kos:
        data += [
            go.Scatter3d(
                x=[0, kos_len if ax=="x" else 0], y=[0, kos_len if ax=="y" else 0], z=[0, kos_len if ax=="z" else 0],
                mode="lines",
                line=dict(color=col, width=10),
                showlegend=False,
                hoverinfo="skip",
            )
            for ax, col in zip(["x", "y", "z"], ["red", "green", "blue"])
        ]
        
    # create figure
    fig = go.Figure(
        data=data,
        layout=dict(
            template="mine_dark+mine_3d",
            scene=dict(aspectmode="data"),
            height=700,
        ),
    )
    return fig



def plot_scrub(snapshots: list[np.ndarray], max_frames: int = 128) -> go.Figure:
    
    n_snapshots = len(snapshots)
    n_points = len(snapshots[0])
    
    # skip snapshots at regular intervals if too many steps given
    framestep = 1
    if n_snapshots > max_frames:
        factor = np.ceil(n_snapshots / max_frames)
        framestep = int(factor)

    # ==========================================================================
    # ANIMATION OPTIONS: how a slider step moves to its frame.
    # This is the 2nd element of a step's `args` (= options for Plotly.animate).
    # ==========================================================================
    anim_opts = dict(
        mode="immediate",        # "immediate": drop queued animations, jump now (good for sliders)
                                 # "next" / "afterall": queue behind the running animation
        frame=dict(
            duration=0,          # ms each frame is shown (matters when playing several frames)
            redraw=True,         # full redraw; REQUIRED for 3D (scene traces can't tween)
        ),
        transition=dict(
            duration=0,          # ms of tweening between frames (only works for some 2D traces)
            easing="linear",     # "cubic-in-out", "elastic", ... (only with duration > 0)
        ),
        fromcurrent=False,       # when playing a list of frames: start at the current one
    )

    fig = go.Figure(
        # ======================================================================
        # BASE DATA: the initial state. Frames patch these traces by index.
        # ======================================================================
        data=[
            # trace 0: the chain (patched by every frame)
            go.Scatter3d(
                x=snapshots[0][:, 0], 
                y=snapshots[0][:, 1], 
                z=snapshots[0][:, 2], 
                mode="markers",
                marker=dict(size=5, color=np.arange(n_points), colorscale="rainbow"),
                name="current",
            ),
        ],

        # ======================================================================
        # FRAMES: precomputed states. Each one is a PATCH, not a replacement:
        # only the given properties change, everything else (color, size...) stays.
        # ======================================================================
        frames=[
            go.Frame(
                name=str(i),     # ID that slider steps refer to in args[0]
                data=[           # patches, paired by position with `traces`
                    go.Scatter3d(
                        x=snapshots[i][:, 0], 
                        y=snapshots[i][:, 1], 
                        z=snapshots[i][:, 2],
                        # styling is kept and not patched!
                    ),
                ],
                traces=[0],      # data[k] patches trace traces[k]; same length as data!
                                 # (if left out, data[k] patches trace k)
                layout=dict(),   # optional layout patch per frame, e.g. title_text=f"point {i}"
                                 # (in 3D, scene camera changes also go here)
            )
            for i in range(0, n_snapshots, framestep)
        ],

        layout=dict(
            template="mine_dark+mine_3d",
            scene_aspectmode="data",
            
            # ==================================================================
            # SLIDER: a list of steps. Each step calls a Plotly.js method with args.
            # ==================================================================
            sliders=[dict(
                active=0,                    # index of the initially selected step
                steps=[
                    dict(
                        label=f"{i:05d}",    # display text on the tick only (no linking role)
                        value=str(i),        # value reported in events; defaults to label
                        method="animate",    # Plotly.js function to call:
                                             #   "animate"  -> go to frames (args below)
                                             #   "restyle"  -> args=[{trace props}, [idx]]
                                             #   "relayout" -> args=[{layout props}]
                                             #   "update"   -> args=[{trace}, {layout}, [idx]]
                                             #   "skip"     -> do nothing (pure UI)
                        args=[
                            [str(i)],        # frame name(s) to go to -> must match Frame.name;
                                             # several names = play them in sequence
                            anim_opts,
                        ],
                        execute=True,        # False: only move the slider, don't call `method`
                    )
                    for i in range(0, n_snapshots, framestep)
                ],

                # current value readout above the slider
                currentvalue=dict(
                    visible=True,
                    prefix="simstep nr: ",
                    suffix="",
                    xanchor="left",          # "left" | "center" | "right"
                    offset=10,               # px between readout and slider
                    font=dict(color="white", size=14),             # e.g. size=14, color="red"
                ),

                # position & size (paper coordinates, 0..1 = plot area)
                x=0, y=0,                    # anchor point; y=0 + yanchor="top" -> BELOW the plot
                xanchor="left", yanchor="top",
                len=1,                       # length as a fraction of the plot width
                pad=dict(t=10, b=10, l=0, r=0),   # px padding around the slider

                # look
                ticklen=0,                   # tick length in px (0 = no ticks)
                minorticklen=0,
                font=dict(color="rgba(0, 0, 0, 0.0)", size=1),                 # tick label font
                # bgcolor, bordercolor, activebgcolor, tickcolor, ... -> colors

                # animation of the slider HANDLE itself (not the plot)
                transition=dict(duration=150, easing="cubic-in-out"),
            )],
        ),
    )

    return fig


