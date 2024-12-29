import numpy as np
import matplotlib
import scipy.stats as stats
from scipy.stats import ttest_ind

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
    :return: List of firing rates per bin (spikes/second).
    """
    # Ensure duration is a scalar
    if isinstance(duration, np.ndarray):
        duration = duration.item()

    # Generate spike train as a binary array (1 for spikes, 0 otherwise)
    num_samples = int(duration * fs)
    spike_train = np.zeros(num_samples)

    for ts in timestamps:
        sample_idx = int(ts * fs)
        if sample_idx < num_samples:
            spike_train[sample_idx] = 1

    # Calculate overall firing rate
    overall_firing_rate = np.sum(spike_train) / duration

    # Calculate firing rate per bin
    bin_samples = int(bin_size * fs)  # Convert bin size to number of samples
    num_bins = len(spike_train) // bin_samples
    firing_rates = []

    for i in range(num_bins):
        bin_start = i * bin_samples
        bin_end = bin_start + bin_samples
        bin_sum = np.sum(spike_train[bin_start:bin_end])
        firing_rate = bin_sum / bin_size
        firing_rates.append(firing_rate)

    return firing_rates


def round_to_half(value):
    """
    Round a value to the nearest half.

    :param value: The value to round.
    :return: Rounded value.
    """
    return round(value * 2) / 2


def independent_t_test(vector1, vector2):
    """
    Perform an independent t-test for two vectors (two-tailed).

    :param vector1: First data vector (length 120).
    :param vector2: Second data vector (length 120).
    :return: Mean of both vectors, t-statistic, p-value, and whether the result is significant (10% level).
    """
    # Ensure both vectors are the same length
    if len(vector1) != len(vector2):
        raise ValueError("The vectors must have the same length.")

    # Perform two-tailed t-test
    t_stat, p_value = stats.ttest_ind(vector1, vector2, alternative='two-sided')

    # Check if the result is significant at 10% level
    is_significant = p_value < 0.025  # Adjusted for one-sided comparison

    mean1 = np.mean(vector1)
    mean2 = np.mean(vector2)
    return mean1, mean2, t_stat, p_value, is_significant


def plot_firing_rate_histogram(firing_rates_file1, firing_rates_file2, title, bin_size=1):
    """
    Plot two histograms of firing rates side-by-side with consistent binning.

    :param firing_rates_file1: Firing rates from the first file.
    :param firing_rates_file2: Firing rates from the second file.
    :param title: Title of the plot.
    :param bin_size: Size of each bin for the histogram.
    """
    # Ensure inputs are numpy arrays
    np.asarray(firing_rates_file1)
    np.asarray(firing_rates_file2)

    # Combine datasets to determine global bin edges
    all_rates = np.concatenate((firing_rates_file1, firing_rates_file2))
    min_rate, max_rate = np.min(all_rates), np.max(all_rates)
    bins = np.arange(min_rate, max_rate + bin_size, bin_size)  # Bin edges

    # Compute histograms for both datasets
    counts_walk, _ = np.histogram(firing_rates_file1, bins=bins)
    counts_stand, _ = np.histogram(firing_rates_file2, bins=bins)

    # Perform t-test and calculate statistics
    mean1, mean2, t_test, p_value, is_significant = independent_t_test(firing_rates_file1, firing_rates_file2)
    print(
        f"t_test: {t_test}, p value: {p_value}, significant: {is_significant}\nMean file1: {mean1:.2f}\nMean file2:"
        f" {mean2:.2f}")

    # Calculate bar positions
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bar_width = (bins[1] - bins[0]) * 0.4  # Bar width as 40% of bin size

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers - bar_width / 2, counts_walk, width=bar_width, color='blue', alpha=0.7, label='file 1',
            edgecolor='black')
    plt.bar(bin_centers + bar_width / 2, counts_stand, width=bar_width, color='red', alpha=0.7, label='file 2',
            edgecolor='black')

    # Add labels and title
    plt.title(title)
    plt.xlabel("Firing Rate (spikes/second)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display t-test results and means on the plot
    plt.text(
        0.95, 0.95,
        f"t={t_test:.2f}\np={p_value:.3f}\nSignificant: {is_significant}\nMean file1: {mean1:.2f}\nMean file2:"
        f" {mean2:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
    )

    # Show the plot
    plt.show()


def find_timestamp_key(mat_data):
    """
    Find the key related to timestamps in the given .mat data.

    :param mat_data: Dictionary containing .mat data.
    :return: Key name for timestamps or None if not found.
    """
    for key in mat_data.keys():
        if key.endswith("_wf_ts") and isinstance(mat_data[key], np.ndarray):
            return key
    return None


if __name__ == "__main__":
    # Load the .mat files
    mat_file_path_file1 = r"data\gonen_ch1_stand.matlab.mat"
    mat_file_path_file2 = r"data\ayelet_ch1_stand.matlab.mat"

    # Load data
    mat_data_file1 = load_mat_file(mat_file_path_file1)
    mat_data_file2 = load_mat_file(mat_file_path_file2)

    # Print keys for inspection
    print_mat_keys(mat_data_file1, name="file1")
    print_mat_keys(mat_data_file2, name="file2")

    # Define timestamp keys
    timestamp_key_file1_a = "SPK01a_wf_ts"
    timestamp_key_file1_b = "SPK01b_wf_ts"
    timestamp_key_file2_a = "SPK01a_wf_ts"
    timestamp_key_file2_b = "SPK01b_wf_ts"

    # Extract timestamps
    timestamps_file1_a = mat_data_file1[timestamp_key_file1_a].flatten()
    timestamps_file1_b = mat_data_file1[timestamp_key_file1_b].flatten()
    timestamps_file2_a = mat_data_file2[timestamp_key_file2_a].flatten()
    timestamps_file2_b = mat_data_file2[timestamp_key_file2_b].flatten()

    # Define parameters
    duration_file1 = mat_data_file1.get("Stop", [0])[0] - mat_data_file1.get("Start", [0])[0]
    duration_file2 = mat_data_file2.get("Stop", [0])[0] - mat_data_file2.get("Start", [0])[0]
    fs = 40000  # Sampling rate in Hz
    bin_size = 1  # Bin size in seconds

    # Compute firing rates for Stand and Walk
    firing_rates_file1_a = firing_rate_with_bins(
        timestamps_file1_a, fs, duration_file1, bin_size, "ch_20 Stand - SPK20a"
    )
    firing_rates_file1_b = firing_rate_with_bins(
        timestamps_file1_b, fs, duration_file1, bin_size, "ch_20 Stand - SPK20b"
    )
    firing_rates_file2_a = firing_rate_with_bins(
        timestamps_file2_a, fs, duration_file2, bin_size, "ch_20 Stand - SPK20a"
    )
    firing_rates_file2_b = firing_rate_with_bins(
        timestamps_file2_b, fs, duration_file2, bin_size, "ch_20 Stand - SPK20b"
    )

    # Plot histogram of firing rates
    plot_firing_rate_histogram(firing_rates_file1_a, firing_rates_file2_a, title="Firing Rate Histogram ch01 Stand a")
