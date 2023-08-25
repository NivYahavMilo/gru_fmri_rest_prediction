import os.path
import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import config

clip_names = ['testretest', 'twomen', 'bridgeville', 'pockets', 'overcome',
              'inception', 'socialnet', 'oceans', 'flower', 'hotel', 'garden', 'dreary', 'homealone', 'brokovich',
              'starwars']


def load_results(area, train_data, test_data):
    res_path = os.path.join(config.RESULTS_DIR, f"{train_data}-training_{test_data}-testing_{area}.pkl")
    with open(res_path, 'rb') as f:
        r = pickle.load(f)

    return r


def compare_temporal_results(areas: list[str], data: dict):
    areas_res = {}
    for area in areas:
        area_data = data[area]['test_mode']['t_test']
        for clip, dynamic in zip(clip_names, area_data.values()):
            areas_res.setdefault(clip, []).append((area, dynamic))

    sns.set_theme(style='darkgrid')
    # Create a plot showing the dynamics of each movie in each region with shaded regions for std
    plt.figure(figsize=(10, 6))

    for movie, regions_dynamic in areas_res.items():
        for region, data in regions_dynamic:
            mean_across_subjects = np.mean(data, axis=0)
            std_across_subjects = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[1])

            plt.plot(mean_across_subjects, label=f'{movie} in {region}')
            plt.fill_between(range(len(mean_across_subjects)),
                             mean_across_subjects - std_across_subjects,
                             mean_across_subjects + std_across_subjects,
                             alpha=0.3)
        plt.legend(loc='upper right')

        sns.set_theme(style='whitegrid')
        plt.title('Dynamics of Movies in Different Regions with Std Deviation')
        plt.xlabel('Timesteps')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()


def compare_mean_results(areas: list[str], data: dict):
    labels = []
    means = []
    stds = []
    rects = []
    colors = ['blue', 'red', 'green']
    sns.set_theme(style='darkgrid')
    fig, ax = plt.subplots()
    for ci, area in enumerate(areas):

        score = data[area]['test_mode']['test']

        means.append(np.mean(score))
        stds.append(np.std(score))
        labels.append(area)

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        rects.append(ax.bar(x - width / 8, means, width, label=area, yerr=stds, capsize=5, color=colors))

    # Add the mean and standard deviation values on top of the bars
    for rect, mean, std, _x in zip(rects, means, stds, x):
        rectangle = rect.patches[_x]
        height = rectangle.get_height()
        ax.annotate(f'{mean:.2f}\n±{std:.2f}',
                    xy=(rectangle.get_x() + rectangle.get_width(), height),
                    xytext=(0, 6),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Mean and Standard Deviation for Each Area')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    legend_handles = [Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]

    ax.legend(handles=legend_handles, loc='upper right')
    plt.show()


def plot_confusion_matrix(areas: list[str], data: dict, title: str, normalize=True):
    sns.set_theme(style='darkgrid')
    for area in areas:
        cm_mtx = data[area]['test_mode']['test_conf_mtx']
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_mtx, display_labels=clip_names)
        cm_display.plot(cmap='Blues')


        plt.xticks(rotation=45)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f"{area} {title}")
        plt.show()


def compare_temporal_networks(train, test):
    res_dict = {}
    networks = ['DMN', 'Visual', 'DorsalAttention']
    for network in networks:
        res_dict[network] = load_results(network, train, test)

    #compare_mean_results(areas=networks, data=res_dict)
    compare_temporal_results(areas=networks, data=res_dict)
    #plot_confusion_matrix(areas=networks, data=res_dict, title=f"Confusion matrix: train-{train}, predict-{test}")

    print()


if __name__ == '__main__':
    compare_temporal_networks(train='clip', test='clip')
