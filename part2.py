import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Change the backend to display plots in a separate window
import matplotlib.pyplot as plt
import scipy.io


def load_mat_file(file_path):
    """
    Load a .mat file and return its content.

    :param file_path: Path to the .mat file.
    :return: A dictionary containing the data from the .mat file.
    """
    return scipy.io.loadmat(file_path)


def print_mat_keys(mat_data, name=""):
    """
    Print the keys (parameters) of the loaded .mat file.

    :param mat_data: Dictionary of the .mat file.
    :param name: Optional name to identify the file.
    """
    print(f"{name} data parameters: {list(mat_data.keys())}")


def extract_waveform_data(mat_data, keyword="wf"):
    """
    Extract waveform-related data from the .mat file based on a keyword.
    Only includes 2D arrays and excludes keys ending with '_ts'.

    :param mat_data: Dictionary of the .mat file.
    :param keyword: Keyword to filter relevant waveform data (default: 'wf').
    :return: Dictionary of relevant waveform data.
    """
    return {
        key: mat_data[key]
        for key in mat_data
        if keyword in key  # Include keys containing the keyword
        and not key.endswith('_ts')  # Exclude keys ending with '_ts'
        and isinstance(mat_data[key], np.ndarray)  # Ensure the value is a numpy array
        and mat_data[key].ndim == 2  # Ensure the array is 2D
    }


def plot_spikes(waveforms_stand, waveforms_walk, stand_title, walk_title, stand_color, walk_color,
                n_samples=None, avg_color='black'):
    """
    Plot waveforms of standing and walking data in the same figure as two subplots.

    :param waveforms_stand: 2D array of standing waveforms.
    :param waveforms_walk: 2D array of walking waveforms.
    :param stand_title: Title for the standing subplot.
    :param walk_title: Title for the walking subplot.
    :param stand_color: Color for the standing waveforms.
    :param walk_color: Color for the walking waveforms.
    :param n_samples: Number of random waveforms to plot (None for all waveforms).
    :param avg_color: Color for the average waveform (default: 'black').
    """
    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot standing waveforms in the first subplot
    num_waveforms_stand = waveforms_stand.shape[1]
    print(f"Number of standing waveforms: {num_waveforms_stand}")
    if n_samples is not None and n_samples < num_waveforms_stand:
        # Randomly sample a subset of waveforms if n_samples is specified
        indices = np.random.choice(num_waveforms_stand, size=n_samples, replace=False)
        waveforms_stand = waveforms_stand[:, indices]

    # Plot each waveform in the standing data
    for i in range(waveforms_stand.shape[1]):
        axs[0].plot(range(waveforms_stand.shape[0]), waveforms_stand[:, i], alpha=0.5, color=stand_color)

    # Calculate and plot the average waveform for standing data
    avg_waveform_stand = np.mean(waveforms_stand, axis=1)
    axs[0].plot(range(waveforms_stand.shape[0]), avg_waveform_stand, color=avg_color, linewidth=2, label='Average Waveform')
    axs[0].set_title(stand_title, fontsize=12)
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[0].grid()

    # Plot walking waveforms in the second subplot
    num_waveforms_walk = waveforms_walk.shape[1]
    print(f"Number of walking waveforms: {num_waveforms_walk}")
    if n_samples is not None and n_samples < num_waveforms_walk:
        # Randomly sample a subset of waveforms if n_samples is specified
        indices = np.random.choice(num_waveforms_walk, size=n_samples, replace=False)
        waveforms_walk = waveforms_walk[:, indices]

    # Plot each waveform in the walking data
    for i in range(waveforms_walk.shape[1]):
        axs[1].plot(range(waveforms_walk.shape[0]), waveforms_walk[:, i], alpha=0.5, color=walk_color)

    # Calculate and plot the average waveform for walking data
    avg_waveform_walk = np.mean(waveforms_walk, axis=1)
    axs[1].plot(range(waveforms_walk.shape[0]), avg_waveform_walk, color=avg_color, linewidth=2, label='Average Waveform')
    axs[1].set_title(walk_title, fontsize=12)
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    axs[1].grid()

    # Adjust the spacing between subplots and show the plot
    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":
    # File paths for the .mat files containing standing and walking data
    mat_file_path_stand = r"data\ayelet_ch1_stand.matlab.mat"
    mat_file_path_walk = r"data\tomer_ch1_walk.matlab.mat"

    # Load the .mat files
    mat_data_stand = load_mat_file(mat_file_path_stand)
    mat_data_walk = load_mat_file(mat_file_path_walk)

    # Print the keys (parameters) of the loaded .mat files
    print_mat_keys(mat_data_walk, name="Walking")
    print_mat_keys(mat_data_stand, name="Standing")

    # Extract waveform data from the loaded .mat files
    waveform_data_stand = extract_waveform_data(mat_data_stand, keyword="wf")
    waveform_data_walk = extract_waveform_data(mat_data_walk, keyword="wf")

    # Plot waveform comparisons if there are relevant keys in the data
    stand_keys = list(waveform_data_stand.keys())
    walk_keys = list(waveform_data_walk.keys())

    if len(stand_keys) > 0 and len(walk_keys) > 0:
        plot_spikes(
            waveforms_stand=waveform_data_stand[stand_keys[0]],
            waveforms_walk=waveform_data_walk[walk_keys[0]],
            stand_title=f"Waveforms of {stand_keys[0]} (stand)",
            walk_title=f"Waveforms of {walk_keys[0]} (walk)",
            stand_color='green',
            walk_color='blue',
            n_samples=100
        )

    if len(stand_keys) > 1 and len(walk_keys) > 1:
        plot_spikes(
            waveforms_stand=waveform_data_stand[stand_keys[1]],
            waveforms_walk=waveform_data_walk[walk_keys[1]],
            stand_title=f"Waveforms of {stand_keys[1]} (stand)",
            walk_title=f"Waveforms of {walk_keys[1]} (walk)",
            stand_color='green',
            walk_color='blue',
            n_samples=100
        )
