import numpy as np 
from matplotlib import pyplot as plt 
import csv
import os
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator

def draw_loss_graph(data):
    if data.shape[1] != 4:
        print("The data doesn't have 4 columns!")
        return
    gen, epoch, l_pi, l_v = data.T
    length = 40
    fig, ax = plt.subplots(2, figsize=(7,8))
    ax[0].plot(l_pi, marker = 'x', color = 'magenta')
    ax[0].set_xlabel("epoch")
    ax[1].plot(l_v, marker = 'x', color = 'cyan')
    ax[1].set_xlabel("epoch")
    for i in range(1, len(gen)):
        if gen[i] > gen[i-1]:
            ax[0].axvline(x=float(i+1), ymin = 0, ymax = 1, color='red', linestyle = '--', linewidth=0.5)
            ax[1].axvline(x=float(i+1), ymin = 0, ymax = 1, color='red', linestyle = '--', linewidth=0.5)
    for a in ax:
        a.grid(True, which='both', linestyle='--', linewidth=0.5)
        
    xticks = np.arange(1, length+1)
    xtick_labels = [int(n) for n in epoch]
    ax[0].set_xticks(xticks, xtick_labels)
    ax[1].set_xticks(xticks, xtick_labels)
    fig.tight_layout()

def add_loss_data(axs, folder, data, color):
    if data.shape[1] != 4:
        print("The data doesn't have 4 columns!")
        return
    gen, epoch, l_pi, l_v = data.T
    
    axs[0].plot(l_pi, ls='dashed', linewidth=1.0, color=color, label=folder)
    axs[1].plot(l_v, ls='dashed', linewidth=1.0, color=color, label=folder)
    return max(gen)

def loss_graph(include, agents_path, epoch:int):
    fig, axs = plt.subplots(2, figsize = (8,10))
    folders = [f for f in os.listdir(agents_path) if os.path.isdir(os.path.join(agents_path, f)) and f in include]
    num_plots = len(folders)
    colormap = cm.get_cmap("rainbow", num_plots)
    gens = []
    for i in range(num_plots):
        folder = folders[i]
        file_path = os.path.join(agents_path, folder, 'loss_record.csv')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)  # Skip the first line
                data = []
                for row in csv_reader:
                    try:
                        row = [float(x) for x in row]
                        data.append(row)
                    except ValueError:
                        continue
                data = np.array(data, dtype=float)
                gens.append(add_loss_data(axs, folder, data, colormap(i)))
    
    max_gen = int(max(gens))
    x_ticks = np.arange(0, (max_gen + 1) * epoch, epoch)
    x_ticklabels = np.arange(0, max_gen + 1)
    x_minorticks = np.arange(0, (max_gen + 1) * epoch, 1)
    
    for ax in axs:
        ax.set_xticks(x_ticks, x_ticklabels)
        ax.set_xticks(x_minorticks, minor=True)
        ax.grid(True, which='major', linestyle='dotted', linewidth=0.3)
        for i in range(1, max_gen + 1):
            ax.axvline(x=float(i*epoch), ymin = 0, ymax = 1, color='black', linewidth=0.5)

    axs[0].set_title("Policy Loss")
    axs[1].set_title("Value Loss")
    axs[1].set_xlabel('generation')
    axs[0].legend(loc='center left', bbox_to_anchor=(0.5, 0.9))
    axs[1].legend(loc='center left', bbox_to_anchor=(0.5, 0.9))
    fig.tight_layout()
                
if __name__ == "__main__":
    agents_path = input("Enter the path to the agents directory: ")
    include = []
    while True:
        folder = input("Enter the folder name to include in the graph (or press Enter to finish): ")
        if folder == "":
            break
        include.append(folder)
    epoch = int(input("Enter the number of epochs per generation: "))
    
    loss_graph(include, agents_path, epoch)
    plt.show()
                