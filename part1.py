import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import firwin, lfilter
import scipy.io

# הגדרת Backend למניעת בעיות גרפיות
matplotlib.use('TkAgg')


# פונקציה ליצירת פילטר High-pass (FIR)
def high_pass_filter_fir(signal, fs, cutoff, numtaps=101):
    """
    פילטר High-pass FIR לסינון תדרים נמוכים.

    :param signal: וקטור הסיגנל.
    :param fs: תדר הדגימה (Hz).
    :param cutoff: תדר החיתוך (Hz).
    :param numtaps: מספר הטאפים (מקדמים) בפילטר FIR (ברירת מחדל: 101).
    :return: הסיגנל המסונן.
    """
    nyquist = 0.5 * fs  # תדר Nyquist
    normal_cutoff = cutoff / nyquist  # תדר חיתוך מנורמל
    fir_coeff = firwin(numtaps, normal_cutoff, pass_zero=False)  # יצירת פילטר High-pass
    filtered_signal = lfilter(fir_coeff, 1.0, signal)  # יישום הפילטר
    return filtered_signal


# פונקציה ליצירת פילטר Low-pass (FIR)
def low_pass_filter_fir(signal, fs, cutoff, numtaps=101):
    """
    פילטר Low-pass FIR לסינון תדרים גבוהים.

    :param signal: וקטור הסיגנל.
    :param fs: תדר הדגימה (Hz).
    :param cutoff: תדר החיתוך (Hz).
    :param numtaps: מספר הטאפים (מקדמים) בפילטר FIR (ברירת מחדל: 101).
    :return: הסיגנל המסונן.
    """
    nyquist = 0.5 * fs  # תדר Nyquist
    normal_cutoff = cutoff / nyquist  # תדר חיתוך מנורמל
    fir_coeff = firwin(numtaps, normal_cutoff, pass_zero=True)  # יצירת פילטר Low-pass
    filtered_signal = lfilter(fir_coeff, 1.0, signal)  # יישום הפילטר
    return filtered_signal


# פונקציה להצגת 3 סיגנלים ב-Subplots
def plot_signals_subplots(time, original_signal, high_passed, low_passed, cutoff_high, cutoff_low):
    """
    מציגה את הסיגנל המקורי, המפולטר High-pass והמפולטר Low-pass ב-Subplots.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Subplot 1: הסיגנל המקורי
    axs[0].plot(time, original_signal, label="Original Signal", color="blue", alpha=0.7)
    axs[0].set_title("Original Signal")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Voltage")
    axs[0].legend(loc="upper right")
    axs[0].grid()
    axs[0].set_xlim(0, 2)

    # Subplot 2: הסיגנל המסונן High-pass
    axs[1].plot(time, high_passed, label=f"High-pass Filtered (cutoff={cutoff_high} Hz)", color="green", linestyle="--")
    axs[1].set_title(f"High-pass Filtered Signal (cutoff={cutoff_high} Hz)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Voltage")
    axs[1].legend(loc="upper right")
    axs[1].grid()
    axs[1].set_xlim(0, 2)

    # Subplot 3: הסיגנל המסונן Low-pass
    axs[2].plot(time, low_passed, label=f"Low-pass Filtered (cutoff={cutoff_low} Hz)", color="red", linestyle=":")
    axs[2].set_title(f"Low-pass Filtered Signal (cutoff={cutoff_low} Hz)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Voltage")
    axs[2].legend(loc="upper right")
    axs[2].grid()
    axs[2].set_xlim(0, 2)

    # ריווח בין הסאבפלוטים
    plt.subplots_adjust(hspace=0.5)

    # הצגת הגרפים
    plt.show()


# פונקציית עיבוד הסיגנל
def process_signal(data, fs):
    """
    מעבדת את הסיגנל המקורי ומחזירה את הסיגנל המקורי, High-pass ו-Low-pass.

    :param data: נתוני הסיגנל (מתח וזמן).
    :param fs: תדר הדגימה (Hz).
    :return: הסיגנל המקורי, הסיגנל המפולטר High-pass, הסיגנל המפולטר Low-pass.
    """
    time = data[:, 1]  # וקטור הזמן
    voltage = data[:, 0]  # וקטור המתח

    # הגדרת תדרי חיתוך
    cutoff_high = 500  # תדר החיתוך ל-High-pass (Hz)
    cutoff_low = 60  # תדר החיתוך ל-Low-pass (Hz)

    # סינון High-pass
    high_passed_signal = high_pass_filter_fir(voltage, fs, cutoff=cutoff_high)

    # סינון Low-pass
    low_passed_signal = low_pass_filter_fir(voltage, fs, cutoff=cutoff_low)

    # החזרת תוצאות
    return time, voltage, high_passed_signal, low_passed_signal, cutoff_high, cutoff_low


if __name__ == "__main__":
    # טעינת הנתונים מקובץ MAT
    mat_file_path = r"data\WB.mat"
    mat_data = scipy.io.loadmat(mat_file_path)
    WB03 = mat_data["WB03"]
    fs = 40000  # Hz
    time, original_signal, high_passed, low_passed, cutoff_high, cutoff_low = process_signal(WB03, fs)
    plot_signals_subplots(time, original_signal, high_passed, low_passed, cutoff_high, cutoff_low)
