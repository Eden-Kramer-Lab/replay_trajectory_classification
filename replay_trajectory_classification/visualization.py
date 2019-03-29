import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase, make_axes

from upsetplot import UpSet

from .analysis import maximum_a_posteriori_estimate
from .classifier import SortedSpikesClassifier


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
    lc.set_linewidth(4)
    line = ax.add_collection(lc)

    # Set the values used for colormapping
    cax, _ = make_axes(ax, location='bottom')
    cbar = ColorbarBase(cax, cmap=cmap, norm=norm,
                        spacing='proportional',
                        orientation='horizontal')
    cbar.set_label('time')

    return line, ax


def plot_all_positions(position_info, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(position_info.x_position.values, position_info.y_position.values,
            color='lightgrey', alpha=0.5, label='all positions')


def make_movie(position, posterior_density, position_info, map_position,
               spikes, place_field_max, movie_name='video_name.mp4'):
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_all_positions(position_info, ax=ax)

    ax.set_xlim(position_info.x_position.min() - 1,
                position_info.x_position.max() + 1)
    ax.set_ylim(position_info.y_position.min() + 1,
                position_info.y_position.max() + 1)
    ax.set_xlabel('x-position')
    ax.set_ylabel('y-position')

    position_dot = plt.scatter([], [], s=80, zorder=102, color='b',
                               label='actual position')
    position_line, = plt.plot([], [], 'b-', linewidth=3)
    map_dot = plt.scatter([], [], s=80, zorder=102, color='r',
                          label='replay position')
    map_line, = plt.plot([], [], 'r-', linewidth=3)
    spikes_dot = plt.scatter([], [], s=40, zorder=104, color='k',
                             label='spikes')
    vmax = np.percentile(posterior_density.values, 99)
    ax.legend()
    posterior_density.isel(time=0).plot(
        x='x_position', y='y_position', vmin=0.0, vmax=vmax,
        ax=ax)
    n_frames = posterior_density.shape[0]

    def _update_plot(time_ind):
        start_ind = max(0, time_ind - 5)
        time_slice = slice(start_ind, time_ind)

        position_dot.set_offsets(position[time_ind])
        position_line.set_data(position[time_slice, 0],
                               position[time_slice, 1])

        map_dot.set_offsets(map_position[time_ind])
        map_line.set_data(map_position[time_slice, 0],
                          map_position[time_slice, 1])

        spikes_dot.set_offsets(place_field_max[spikes[time_ind] > 0])

        im = posterior_density.isel(time=time_ind).plot(
            x='x_position', y='y_position', vmin=0.0, vmax=vmax,
            ax=ax, add_colorbar=False)

        return position_dot, im

    movie = animation.FuncAnimation(fig, _update_plot, frames=n_frames,
                                    interval=50, blit=True)
    if movie_name is not None:
        movie.save(movie_name, writer=writer)

    return fig, movie


def plot_ripple_decode(ripple_number, results, ripple_position,
                       ripple_spikes, position, linear_position_order):
    result = results.sel(ripple_number=ripple_number).assign_coords(
        time=lambda ds: ds.time / np.timedelta64(1, 's'),
    )
    time = result.time.values
    map_estimate = maximum_a_posteriori_estimate(
        result.acausal_posterior.sum('state'))
    spike_time_ind, neuron_ind = np.nonzero(
        ripple_spikes.loc[ripple_number].values[:, linear_position_order])
    n_neurons = ripple_spikes.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    g = SortedSpikesClassifier.predict_proba(result).acausal_posterior.plot(
        hue='state', ax=axes[0], linewidth=2)
    axes[0].set_ylim((0, 1))

    twin_ax = axes[0].twinx()
    twin_ax.scatter(time[spike_time_ind], neuron_ind, color='black', zorder=1,
                    marker='|', s=80, linewidth=3)
    twin_ax.set_ylim((-0.5, n_neurons - 0.5))
    twin_ax.set_ylabel('Neuron ID')

    box = axes[0].get_position()
    axes[0].set_position([box.x0, box.y0 + box.height * 0.1,
                          box.width, box.height * 0.9])

    axes[0].legend(g, result.state.values, loc='upper right',
                   bbox_to_anchor=(1.0, -0.05), fancybox=False, shadow=False,
                   ncol=1, frameon=False)

    axes[1].plot(position.values[:, 0], position.values[:, 1],
                 color='lightgrey', alpha=0.4, zorder=0)
    plot_2D_position_with_color_time(
        time, map_estimate, ax=axes[1])
    axes[1].scatter(ripple_position.loc[ripple_number].values[:, 0],
                    ripple_position.loc[ripple_number].values[:, 1],
                    color='black', s=100, label='actual position')
    result.sum(['state', 'time']).acausal_posterior.plot(
        x='x_position', y='y_position', robust=True, cmap='Purples', alpha=0.3,
        ax=axes[1], add_colorbar=False, zorder=0)

    axes[1].legend()


def plot_neuron_place_field_2D_1D_position(
        position_info, place_field_max, linear_place_field_max,
        linear_position_order):
    position = np.asarray(position_info.loc[:, ['x_position', 'y_position']])
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    axes[0].plot(position[:, 0], position[:, 1], color='lightgrey', alpha=0.4)
    axes[0].set_ylabel('y-position')
    axes[0].set_xlabel('x-position')
    zipped = zip(linear_place_field_max[linear_position_order],
                 place_field_max[linear_position_order])
    for ind, (linear, place_max) in enumerate(zipped):
        axes[0].scatter(place_max[0], place_max[1], s=300, alpha=0.3)
        axes[0].text(place_max[0], place_max[1], linear_position_order[ind],
                     fontsize=15, horizontalalignment='center',
                     verticalalignment='center')
        axes[1].scatter(ind, linear, s=200, alpha=0.3)
        axes[1].text(ind, linear, linear_position_order[ind], fontsize=15,
                     horizontalalignment='center', verticalalignment='center')
    max_df = (position_info.groupby('arm_name').linear_position2.max()
              .iteritems())
    for arm_name, max_position in max_df:
        axes[1].axhline(max_position, color='lightgrey', zorder=0,
                        linestyle='--')
        axes[1].text(0, max_position - 0.2, arm_name, color='lightgrey',
                     horizontalalignment='left', verticalalignment='top',
                     fontsize=12)
    axes[1].set_ylim((-3.0, position_info.linear_position2.max() + 3.0))
    axes[1].set_ylabel('linear position')
    axes[1].set_xlabel('Neuron ID')


def plot_category_counts(replay_info):
    upset = UpSet(replay_info.set_index(['hover', 'continuous', 'fragmented']),
                  sum_over=False, intersection_plot_elements=3)
    return upset.plot()


def plot_category_duration(replay_info):
    is_duration_col = replay_info.columns.str.endswith('_duration')
    sns.stripplot(data=(replay_info.loc[:, is_duration_col]
                        .rename(columns=lambda c: c.split('_')[0])),
                  order=['continuous', 'fragmented', 'hover'],
                  orient='horizontal')
    plt.xlabel('Duration (s)')
    sns.despine(left=True)
