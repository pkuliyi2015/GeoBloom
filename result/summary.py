'''
This script is used to summarize the results of the repeated experiments of GeoBloom
We need to report the average and standard deviation of the Recall@20, Recall@10, NDCG@5, and NDCG@1 and the average best dev time, std of best dev time.
'''

import re
import numpy as np

def parse_results(file_content):
    # Regular expression to extract metrics
    pattern = r"Recall@20: (\d+\.\d+)\s+Recall@10: (\d+\.\d+)\s+NDCG@5: (\d+\.\d+)\s+NDCG@1: (\d+\.\d+)\s+Best dev time: (\d+\.\d+)"
    matches = re.findall(pattern, file_content)

    # Convert matches to numpy array for easy calculation
    data = np.array(matches, dtype=float)
    return data

def calculate_statistics(data):
    # Calculate mean and standard deviation
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return means, stds

def summarize_results(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()

    data = parse_results(file_content)
    means, stds = calculate_statistics(data)

    # Print the results
    print("Average Recall@20: {:.6f}, Std: {:.6f}".format(means[0], stds[0]))
    print("Average Recall@10: {:.6f}, Std: {:.6f}".format(means[1], stds[1]))
    print("Average NDCG@5: {:.6f}, Std: {:.6f}".format(means[2], stds[2]))
    print("Average NDCG@1: {:.6f}, Std: {:.6f}".format(means[3], stds[3]))
    
    # Convert average best dev time from seconds to hours, minutes, and seconds
    avg_time_hours = int(means[4] // 3600)
    avg_time_minutes = int((means[4] % 3600) // 60)
    avg_time_seconds = means[4] % 60

    std_time_hours = int(stds[4] // 3600)
    std_time_minutes = int((stds[4] % 3600) // 60)
    std_time_seconds = stds[4] % 60

    print("Average Best dev time: {}h {}m {:.2f}s, Std: {}h {}m {:.2f}s".format(
        avg_time_hours, avg_time_minutes, avg_time_seconds,
        std_time_hours, std_time_minutes, std_time_seconds))

path = 'result/'
dataset_names = ['Beijing', 'Shanghai', 'GeoGLUE', 'GeoGLUE_clean']
version = '19'

for dataset_name in dataset_names:
    file_path = f'{path}{dataset_name}_v{version}_test.txt'
    print(f'Summarizing results for {dataset_name} v{version}')
    summarize_results(file_path)
