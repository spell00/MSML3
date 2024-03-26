import csv
import time
import os
import multiprocessing
from pyteomics import mzml

total_time = time.time()
class ConvertMzmlToCsv:
    def __init__(self, input_folder, output_folder, intensity_threshold, filenames):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.intensity_threshold = intensity_threshold
        self.filenames = filenames

    def convert(self, i):
        t = time.time()
        filename = self.filenames[i]
        print(f"Converting {filename} to CSV...")
        input_mzml_file = os.path.join(self.input_folder, filename)
        output_csv_file = os.path.join(self.output_folder, f'{os.path.splitext(filename)[0]}.csv')

        # Open the mzML file for reading
        with mzml.read(input_mzml_file) as mzml_file:
            # Create a CSV file for writing
            with open(output_csv_file, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                
                # Write the header row
                csv_writer.writerow(['ScanNumber', 'RetentionTime', 'm/z', 'Intensity'])

                # Iterate through spectra in the mzML file
                for spectrum in mzml_file:
                    scan_number = spectrum['id']
                    retention_time = spectrum['scanList']['scan'][0]['scan start time']
                    mzs = spectrum['m/z array']
                    intensities = spectrum['intensity array']

                    # Filter data based on intensity threshold
                    filtered_data = [(mz, intensity) for mz, intensity in zip(mzs, intensities) if intensity >= self.intensity_threshold]

                    # Write the filtered data to the CSV file
                    for mz, intensity in filtered_data:
                        csv_writer.writerow([scan_number, retention_time, mz, intensity])
        hours = int((time.time() - t) // 3600)
        minutes = int((time.time() - t) % 3600) // 60
        seconds = int((time.time() - t) % 60)
        print(f"Conversion completed for {filename}. Data saved to {output_csv_file} in {hours} hours, {minutes} minutes, and {seconds} seconds")


def main(input_folder, output_folder, intensity_threshold):
    filenames = os.listdir(input_folder)
    # Ensure the output folder exists, or create it if it doesn't
    os.makedirs(output_folder, exist_ok=True)

    n_cpus = multiprocessing.cpu_count()/2
    pool = multiprocessing.Pool(int(n_cpus))
    fun = ConvertMzmlToCsv(input_folder, output_folder, intensity_threshold, filenames)
    pool.map(fun.convert, range(len(filenames)))
    pool.close()
    pool.join()

if __name__ == "__main__":

    # Specify the input folder containing mzML files and the output folder for CSV files
    input_folder = 'resources/bacteries_2024/02-02-2024/mzml'  # Replace with the actual folder path containing mzML files
    output_folder = 'resources/bacteries_2024/02-02-2024/csv'  # Replace with the desired folder path for CSV files

    # Intensity filter threshold
    intensity_threshold = 1000

    main(input_folder, output_folder, intensity_threshold)