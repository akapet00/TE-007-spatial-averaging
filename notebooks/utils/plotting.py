import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


__all__ = ['set_3d_params', 'set_axes_equal', 'config_rcparams']


def set_3d_params(ax, aspect=[1, 1, 1]):
    """Configure pane and ticks for 3D plots.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        Axes subplot
    aspect : list, optional
        Axes aspect ratio

    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        Adjusted axes subplot
    """
    ax.set(xlabel='x',
           ylabel='y',
           zlabel='z')
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_box_aspect(aspect)
    return ax


def config_rcparams(reset=False):
    """Set basic configuration for plotting within notebooks.
    
    Parameters
    ----------
    reset : bool, optional  
        Reset to default rc parameter values
    
    Returns
    -------
    None
    """
    if reset:
        sns.reset_defaults()
    else:
        sns.set(context='notebook',
                style='white',
                rc={'xtick.bottom': True,
                    'xtick.color': 'black',
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',})
        sns.set_palette('colorblind')


def set_axes_equal(ax):
    """Return 3-D axes with equal scale.

    Note: This function is implemented as in:
    https://stackoverflow.com/a/31364297/15005103 because there is no
    support setting that would enable `ax.axis('equal')` in 3D.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3-D axes subplot with scale settings set to `auto`

    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        Axes as if the scale settings were defined as `equal`
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # bounding box is a sphere in the sense of the infinity norm
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return ax
