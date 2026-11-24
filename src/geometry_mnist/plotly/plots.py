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
