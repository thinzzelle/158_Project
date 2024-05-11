import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import gaussian_kde


def read_data(file_path):
    matching_pair_list = []
    non_matching_pair_list = []

    with open(file_path, 'r') as file:
        file.readline()  # Skip the header
        count = 0
        for line in file:
            parts = line.split()
            # Extracting and converting the distance to float
            distance = float(parts[7][:7])
            if parts[0] == parts[2]:
                matching_pair_list.append(distance)
            else:
                non_matching_pair_list.append(distance)

            count += 1
            if count > 10000:
                break

    return matching_pair_list, non_matching_pair_list


def plot_graph_density(matching_pair_list, non_matching_pair_list, model, race):
    density_matching = gaussian_kde(matching_pair_list)
    density_non_matching = gaussian_kde(non_matching_pair_list)

    print(density_matching)
    print(density_non_matching)

    x = np.linspace(0, 1.2, 1000)

    # Plot the density curves
    plt.plot(x, density_matching(x), label='Positive Pair', color='blue')
    plt.plot(x, density_non_matching(x), label='Negative Pair', color='orange')

    # Add labels and legend
    plt.xlabel('Cosine Distance')
    plt.ylabel('Density')
    plt.title(f'{model}, {race}')
    plt.legend()

    # Show plot
    plt.show()


def main():
    model_list = ['DeepFace', 'ArcFace', 'Facenet', 'Facenet512']
    # model_list = ['Facenet']
    race_list = ['African', 'Asian', 'Caucasian', 'Indian']

    for model in model_list:
        for race in race_list:
            file_path = f'testing_results/verification/{model}/{race}_results.txt'
            matching_pair_list, non_matching_pair_list = read_data(file_path)
            plot_graph_density(matching_pair_list,
                               non_matching_pair_list, model, race)


if __name__ == "__main__":
    main()
