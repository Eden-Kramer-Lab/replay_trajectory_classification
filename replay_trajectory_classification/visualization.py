import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase, make_axes
from matplotlib.colors import ListedColormap


def plot_replay_spiking_ordered_by_place_fields(
        spikes, place_field_firing_rates, place_bin_centers,
        ax=None, cmap=None, sampling_frequency=1, time=None):
    '''Plot spikes by the positiion of their maximum place field firing rate.

    Parameters
    ----------
    spikes : ndarray, shape (n_time, n_neurons)
    place_field_firing_rates : ndarray, shape (n_neurons, n_place_bins)
    place_bin_centers : ndarray, shape (n_place_bins,)
    ax : None or matplotlib axis, optional
    cmap : None, str, or array, optional
    sampling_frequency : float, optional
    time : ndarray, shape (n_time,), optional

    Returns
    -------
    ax : matplotlib axis
    im : scatter plot handle

    '''
    ax = ax or plt.gca()
    AVG_PLACE_FIELD_SIZE = 25
    n_colors = int(np.ceil(np.ptp(place_bin_centers) / AVG_PLACE_FIELD_SIZE))
    cmap = cmap or ListedColormap(sns.color_palette('hls', n_colors))

    n_time, n_neurons = spikes.shape
    if time is None:
        time = np.arange(n_time) / sampling_frequency

    cmap = plt.get_cmap(cmap)
    neuron_to_place_bin = np.argmax(place_field_firing_rates, axis=1)
    ordered_place_field_to_neuron = np.argsort(neuron_to_place_bin)
    neuron_to_ordered_place_field = np.argsort(ordered_place_field_to_neuron)

    time_ind, neuron_ind = np.nonzero(spikes)
    im = ax.scatter(time[time_ind], neuron_to_ordered_place_field[neuron_ind],
                    c=place_bin_centers[neuron_to_place_bin[neuron_ind]],
                    cmap=cmap, vmin=np.floor(place_bin_centers.min()),
                    vmax=np.ceil(place_bin_centers.max()))
    plt.colorbar(im, ax=ax, label='position')

    ax.set_xlim(time[[0, -1]])
    ax.set_xlabel('Time')
    ax.set_ylabel('Neurons')
    ax.set_yticks(np.arange(n_neurons))
    ax.set_yticklabels(ordered_place_field_to_neuron + 1)
    ax.set_ylim((-0.25, n_neurons + 1.00))
    sns.despine()

    return ax, im


def plot_2D_position_with_color_time(time, position, ax=None, cmap='plasma',
                                     alpha=None):
    '''

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    position : ndarray, shape (n_time, 2)
    ax : None or `matplotlib.axes.Axes` instance
    cmap : str
    alpha : None or ndarray, shape (n_time,)

    Returns
    -------
    line : `matplotlib.collections.LineCollection` instance
    ax : `matplotlib.axes.Axes` instance

    '''
    if ax is None:
        ax = plt.gca()
    points = position.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(time.min(), time.max())
    cmap = plt.get_cmap(cmap)
    colors = cmap(norm(time))
    if alpha is not None:
        colors[:, -1] = alpha
    lc = LineCollection(segments, colors=colors, zorder=100)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)

    # Set the values used for colormapping
    cax, _ = make_axes(ax, location='bottom')
    cbar = ColorbarBase(cax, cmap=cmap, norm=norm,
                        spacing='proportional',
                        orientation='horizontal')
    cbar.set_label('time')

    return line, ax


def maximum_a_posteriori_estimate(posterior_density):
    '''

    Parameters
    ----------
    posterior_density : xarray.DataArray, shape (n_time, n_x_bins, n_y_bins)

    Returns
    -------
    map_estimate : ndarray, shape (n_time,)

    '''
    stacked_posterior = np.log(posterior_density.stack(
        z=['x_position', 'y_position']))
    map_estimate = stacked_posterior.z[stacked_posterior.argmax('z')]
    return np.asarray(map_estimate.values.tolist())
