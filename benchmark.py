import subprocess
import time
import matplotlib.pyplot as plt
import csv
import shutil
import os

def run_script(script_name):
    """Triggers the script to run and calculates the time it takes to complete the process"""
    start_time = time.time()
    subprocess.run(["python", script_name])
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def clear_output_folder():
    """Deletes the contents of the output folder before starting each benchmarking run"""
    output_folder = './output'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

def main():
    """Defines the nr of benchmarking cycles (num_runs) and plots the results (also saves them as .csv)"""
    num_runs = 5
    auto_plate_times = []
    parallel_plate_times = []

    for _ in range(num_runs):
        clear_output_folder()

        parallel_plate_time = run_script("parallel_plate_plotter.py")
        auto_plate_time = run_script("auto_plate_plotter.py")
        
        parallel_plate_times.append(parallel_plate_time)
        auto_plate_times.append(auto_plate_time)
        

    # Save results to CSV
    with open('script_execution_times.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Run', 'auto_plate_plotter.py', 'parallel_plate_plotter.py'])
        for i in range(num_runs):
            csv_writer.writerow([i + 1, auto_plate_times[i], parallel_plate_times[i]])

    # Plotting
    plt.scatter(range(1, num_runs + 1), auto_plate_times, label='auto_plate_plotter.py')
    plt.scatter(range(1, num_runs + 1), parallel_plate_times, label='parallel_plate_plotter.py')
    plt.xlabel('Run Number')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('Script Execution Time Comparison')
    plt.show()

if __name__ == "__main__":
    main()