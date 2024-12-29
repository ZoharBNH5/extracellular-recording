import numpy as np
import matplotlib

# Set the backend for matplotlib to display plots in a separate window
matplotlib.use('TkAgg')
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


def firing_rate_with_bins(timestamps, fs, duration, bin_size, file_name):
    """
    Compute the spike train, overall firing rate, and firing rate in bins.

    :param timestamps: Array of spike timestamps (in seconds).
    :param fs: Sampling rate (in Hz).
    :param duration: Total recording duration (in seconds).
    :param bin_size: Size of each bin (in seconds).
    :param file_name: Name of the file being processed (used in the plot title).
    :return: List of firing rates per bin (spikes/second).
    """
    # Ensure duration is a scalar in case it's provided as a numpy array
    if isinstance(duration, np.ndarray):
        duration = duration.item()

    # Generate spike train as a binary array (1 for spikes, 0 otherwise)
    num_samples = int(duration * fs)
    spike_train = np.zeros(num_samples)

    # Convert timestamps to spike train
    for ts in timestamps:
        sample_idx = int(ts * fs)
        if sample_idx < num_samples:  # Avoid indexing beyond the array
            spike_train[sample_idx] = 1

    # Calculate overall firing rate (spikes per second)
    overall_firing_rate = np.sum(spike_train) / duration

    # Calculate firing rates for each bin
    bin_samples = int(bin_size * fs)  # Number of samples in each bin
    num_bins = len(spike_train) // bin_samples
    firing_rates = []

    # Loop through each bin and calculate the firing rate
    for i in range(num_bins):
        bin_start = i * bin_samples
        bin_end = bin_start + bin_samples
        bin_sum = np.sum(spike_train[bin_start:bin_end])  # Count spikes in the bin
        firing_rate = bin_sum / bin_size  # Convert to rate (spikes/second)
        firing_rates.append(firing_rate)

    # Plot the firing rate as a function of time
    time_bins = np.arange(num_bins) * bin_size  # Time corresponding to each bin
    plt.figure(figsize=(12, 6))
    plt.plot(time_bins, firing_rates, linestyle='-', color='blue', label='Firing Rate')
    plt.title(f"Firing Rate as a Function of Time ({file_name})")
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (spikes/second)")
    plt.grid()
    plt.legend()
    plt.show()

    return firing_rates


def round_to_half(value):
    """
    Round a value to the nearest half.

    :param value: The value to round.
    :return: Rounded value.
    """
    return round(value * 2) / 2


def find_timestamp_key(mat_data):
    """
    Find the key related to timestamps in the given .mat data.

    :param mat_data: Dictionary containing .mat data.
    :return: Key name for timestamps or None if not found.
    """
    for key in mat_data.keys():
        # Look for keys ending with "_wf_ts" that are numpy arrays
        if key.endswith("_wf_ts") and isinstance(mat_data[key], np.ndarray):
            return key
    return None


if __name__ == "__main__":
    # File paths for the .mat files containing Stand and Walk data
    mat_file_path_stand = r"data\gonen_ch20_stand.matlab.mat"
    mat_file_path_walk = r"data\maayan_ch20_walk.matlab.mat"

    # Load the .mat files into dictionaries
    mat_data_stand = load_mat_file(mat_file_path_stand)
    mat_data_walk = load_mat_file(mat_file_path_walk)

    # Print the keys (parameters) of the loaded .mat files for inspection
    print_mat_keys(mat_data_walk, name="Walk")
    print_mat_keys(mat_data_stand, name="Stand")

    # Define keys for timestamps in the .mat data
    timestamp_key_stand_a = "SPK20a_wf_ts"  # Timestamps for Stand condition (channel a)
    timestamp_key_stand_b = "SPK20b_wf_ts"  # Timestamps for Stand condition (channel b)
    timestamp_key_walk_a = "SPK20a_wf_ts"  # Timestamps for Walk condition (channel a)
    timestamp_key_walk_b = "SPK20b_wf_ts"  # Timestamps for Walk condition (channel b)

    # Extract timestamps from the .mat data
    timestamps_stand_a = mat_data_stand[timestamp_key_stand_a].flatten()
    timestamps_stand_b = mat_data_stand[timestamp_key_stand_b].flatten()
    timestamps_walk_a = mat_data_walk[timestamp_key_walk_a].flatten()
    timestamps_walk_b = mat_data_walk[timestamp_key_walk_b].flatten()

    # Define recording durations and sampling rate
    duration_stand = mat_data_stand.get("Stop", [0])[0] - mat_data_stand.get("Start", [0])[0]
    duration_walk = mat_data_walk.get("Stop", [0])[0] - mat_data_walk.get("Start", [0])[0]
    fs = 40000  # Sampling rate in Hz
    bin_size = 1  # Bin size in seconds

    # Compute firing rates for Stand and Walk conditions (channel a and b)
    firing_rates_stand_a = firing_rate_with_bins(
        timestamps_stand_a, fs, duration_stand, bin_size, "ch_20 Stand - SPK20a"
    )
    firing_rates_stand_b = firing_rate_with_bins(
        timestamps_stand_b, fs, duration_stand, bin_size, "ch_20 Stand - SPK20b"
    )
    firing_rates_walk_a = firing_rate_with_bins(
        timestamps_walk_a, fs, duration_walk, bin_size, "ch_20 Walk - SPK20a"
    )
    firing_rates_walk_b = firing_rate_with_bins(
        timestamps_walk_b, fs, duration_walk, bin_size, "ch_20 Walk - SPK20b"
    )
