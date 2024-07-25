import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_directory(directory, file_type):
    # Collect file information
    files_info = []
    for file in os.listdir(f'{directory}/{file_type}'):
        file_path = os.path.join(directory, file_type, file)
        file_size = os.path.getsize(file_path) / (1024**2)  # Convert size to MB
        file_extension = os.path.splitext(file)[1].lower()
        files_info.append((file, file_size, file_extension))
    
    # Create a DataFrame
    df = pd.DataFrame(files_info, columns=['Filename', 'Size (MB)', 'Extension'])
    
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
    plt.grid(True)
    os.makedirs(f"{directory}/summary/{file_type}", exist_ok=True)
    plt.savefig(f"{directory}/summary/{file_type}/hist.png")
    plt.close()

# Replace 'your_directory_path' with the actual path to the directory you want to analyze
directory_path = 'resources/bacteries_2024/B15-06-29-2024'
os.makedirs(f"{directory_path}/summary", exist_ok=True)
file_types = ['raw', 'mzdb/200spd', 'tsv/mz10/rt10/200spd/ms2/all', 'tsv/mz0.1/rt10/200spd/ms2/all']

for file_type in file_types:
    analyze_directory(directory_path, file_type)
