import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_directory(directory, file_type):
    # Collect file information
    if 'tsv' in file_type:
        file_type += '/all'
    files_info = []
    for file in os.listdir(f'{directory}/{file_type}'):
        file_path = os.path.join(directory, file_type, file)
        file_size = os.path.getsize(file_path) / (1024**2)  # Convert size to MB
        file_extension = os.path.splitext(file)[1].lower()
        files_info.append((file, file_size, file_extension))
    
    # Create a DataFrame
    df = pd.DataFrame(files_info, columns=['Filename', 'Size (MB)', 'Extension'])
    
    if 'tsv' in file_type:
        file_type = file_type.replace('/all', '')
    # Basic statistics
    total_files = len(df)
    total_size = df['Size (MB)'].sum()
    avg_size = df['Size (MB)'].mean()
    largest_file = df.loc[df['Size (MB)'].idxmax()]
    smallest_file = df.loc[df['Size (MB)'].idxmin()]

    # Print basic statistics
    print(f'--- {file_type} ---')
    print(f'Total number of files: {total_files}')
    print(f'Total size of files: {total_size:.2f} MB')
    print(f'Average file size: {avg_size:.2f} MB')
    print(f'Largest file: {largest_file["Filename"]} ({largest_file["Size (MB)"]:.2f} MB)')
    print(f'Smallest file: {smallest_file["Filename"]} ({smallest_file["Size (MB)"]:.2f} MB)')

    # Plot histogram of file sizes
    plt.figure(figsize=(10, 6))
    plt.hist(df['Size (MB)'], bins=50, color='blue', edgecolor='black')
    plt.title(f'Histogram of File Sizes for {file_type}')
    plt.xlabel('File Size (MB)')
    plt.ylabel('Number of Files')
    os.makedirs(f"{directory}/summary/{file_type}", exist_ok=True)
    plt.rc('font', size=16)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=16)
    
    plt.savefig(f"{directory}/summary/{file_type}/hist.png")
    # Increase all font sizes
    plt.close()

    # Plot boxplot of file sizes
    # Boxplot is useful for visualizing the spread and skewness of the data
    # Use seaborn for a more visually appealing boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Size (MB)'], color='orange')
    plt.title(f'Boxplot of File Sizes for {file_type}')
    plt.xlabel('File Size (MB)')

    plt.rc('font', size=16)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=16)
    
    plt.savefig(f"{directory}/summary/{file_type}/boxplot.png")
    plt.close()


def analyze_processing_time(directory, file_type='raw'):
    if file_type == 'raw':
        df = pd.read_csv(f'{directory}/processing_times_raw.log')
        # Remove rows with missing values
        df = df.dropna()
        # Drop rows with processing time less than 0
        df = df[df['duration(s)'] >= 0]

        total_processing_time = df['duration(s)'].sum()
        avg_processing_time = df['duration(s)'].mean()
        largest_processing_time = df.loc[df['duration(s)'].idxmax()]
        smallest_processing_time = df.loc[df['duration(s)'].idxmin()]
        colname = 'duration(s)'
    elif file_type == 'mzdb/200spd':
        df = pd.read_csv(f'msml/mzdb2tsv/processing_times.csv')
        # Remove rows with missing values
        df = df.dropna()
        # Drop rows with processing time less than 0
        df = df[df['time(s)'] >= 0]

        total_processing_time = df['time(s)'].sum()
        avg_processing_time = df['time(s)'].mean()
        largest_processing_time = df.loc[df['time(s)'].idxmax()]
        smallest_processing_time = df.loc[df['time(s)'].idxmin()]
        colname = 'time(s)'
    elif file_type.__contains__('tsv'):
        df = pd.read_csv(f'{directory}/{file_type}/time.csv')
        # Remove rows with missing values
        df = df.dropna()
        # Drop rows with processing time less than 0
        df = df[df['Time'] >= 0]

        total_processing_time = df['Time'].sum()
        avg_processing_time = df['Time'].mean()
        largest_processing_time = df.loc[df['Time'].idxmax()]
        smallest_processing_time = df.loc[df['Time'].idxmin()]
        colname = 'Time'
    else:
        raise ValueError('Invalid file type. Please choose from: raw, mzdb/200spd, tsv/mz10/rt10/200spd/ms2/all, tsv/mz0.1/rt10/200spd/ms2/all')

    print(f'Total processing time: {total_processing_time:.2f} s')
    print(f'Average processing time: {avg_processing_time:.2f} s')
    print(f'Largest processing time: {largest_processing_time[colname]} ({largest_processing_time[colname]:.2f} s)')
    print(f'Smallest processing time: {smallest_processing_time[colname]} ({smallest_processing_time[colname]:.2f} s)')
    

    # Plot histogram of processing times
    plt.figure(figsize=(10, 6))
    plt.hist(df[colname], bins=50, color='green', edgecolor='black')
    plt.title(f'Histogram of Processing Times for {file_type}')
    plt.xlabel('Processing Time (s)')
    plt.ylabel('Number of Files')
    plt.savefig(f"{directory}/summary/{file_type}/processing_time_hist.png")
    plt.close()

    # Plot boxplot of processing times
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[colname], color='red')
    plt.title(f'Boxplot of Processing Times for {file_type}')
    plt.xlabel('Processing Time (s)')
    plt.savefig(f"{directory}/summary/{file_type}/processing_time_boxplot.png")
    plt.close()

if __name__ == '__main__':

    # Replace 'your_directory_path' with the actual path to the directory you want to analyze
    directory_path = 'resources/bacteries_2024/B15-06-29-2024'
    os.makedirs(f"{directory_path}/summary", exist_ok=True)
    file_types = ['raw', 'mzdb/200spd', 'tsv/mz10/rt10/200spd/ms2']

    for file_type in file_types:
        analyze_directory(directory_path, file_type)
        analyze_processing_time(directory_path, file_type)
        
    # analyze_processing_time(directory_path)

    tsv2df_path = f'resources/bacteries_2024/matrices/mz10/rt10/thr0.0/'\
                    f'200spd/ms2/combat0/shift0/none/loginloop/mutual_info_classif/all'\
                    f'/all_B15_gkf0_mz0-10000rt0-320_5splits'