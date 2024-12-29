# Extracellular Recording

This project focuses on analyzing extracellular neural recordings, extracting relevant features, and visualizing the results. The analysis includes spike train computation, firing rate calculation, and statistical comparisons between conditions such as standing and walking.

---

## How to Run the Project

1. **Prepare the Data**:
   - Place your `.mat` files containing spike timestamps and metadata in the `data/` folder.

2. **Install Dependencies**:
   - Ensure you have Python installed along with the following libraries:
     ```bash
     pip install numpy matplotlib scipy
     ```

3. **Run the Scripts**:
   - To analyze your data, use the provided Python scripts:
     - `part1.py`: For basic spike train and firing rate computation.
     - `part2.py`: For statistical analysis (e.g., t-tests between conditions).
     - `part3.py`: For advanced visualizations such as firing rate histograms.
     - `par3_4.py`: For specific and custom analyses related to spike data.

4. **Example Command**:
   - To compute and visualize firing rates:
     ```bash
     python part1.py
     ```

5. **Output**:
   - The scripts generate plots, firing rate data, and statistical results depending on the analysis performed.

---

## Repository Structure

The project contains the following files and folders:
```plaintext
extracellular-recording/
├── data/                   # Folder for all data files (.mat and .nex)
│   ├── WB.mat
│   ├── ayelet_ch1_stand.matlab.mat
│   ├── maayan_ch1_walk.matlab.mat
│   ├── gonen_ch20_stand.matlab.mat
│   └── ...                 # Additional .mat and .nex files
├── par3_4.py               # Script for advanced spike data analysis
├── part1.py                # Utilities for spike train and firing rate computation
├── part2.py                # Statistical analysis and comparisons (e.g., t-tests)
├── part3.py                # Visualization tools for advanced data plotting
├── README.md               # Project documentation
└── .idea/                  # PyCharm project configuration files

```

---

### Notes

- Ensure the file paths in the scripts point to the correct `.mat` files in the `data/` folder.
- If you encounter issues with missing libraries, install them using `pip` as shown above.

---

## Future Work

- Extend the functionality to support additional file formats like HDF5 and CSV.
- Develop machine learning models for spike sorting.
- Add interactive visualization tools.

---

## Contributors

- **ZoharBNH5** (Main Developer)

Feel free to contribute by creating issues or submitting pull requests to enhance this project!
