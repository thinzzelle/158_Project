import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


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


def calculate_rate_and_threshold(positive_pairs, negative_pairs, thresholds):
    fmr_list = []
    fnmr_list = []
    with open('tmp/temporary.txt', 'w') as file:
        pass

    min_value = 1
    for threshold in thresholds:
        false_positive = sum(
            1 for distance in positive_pairs if distance > threshold)
        true_negative = len(positive_pairs) - false_positive

        false_negative = sum(
            1 for distance in negative_pairs if distance < threshold)
        true_positive = len(negative_pairs) - false_negative

        fmr = false_positive / \
            (false_positive + true_negative) if (false_positive +
                                                 true_negative) != 0 else 0
        fnmr = false_negative / \
            (false_negative + true_positive) if (true_positive +
                                                 false_negative) != 0 else 0

        # find the minimum value of fmr and fnmr
        if fmr >= fnmr and fmr < min_value:
            min_value = fmr
            optimal_threshold = threshold
        elif fnmr > fmr and fnmr < min_value:
            min_value = fnmr
            optimal_threshold = threshold

        fmr_list.append(fmr)
        fnmr_list.append(fnmr)

    return fmr_list, fnmr_list, optimal_threshold


def calculate_accuracy(matrix):
    return (matrix[0] + matrix[3]) / sum(matrix)


def plot_rates(model, race_list, fmr_dict, fnmr_dict, optimal_threshold_dict, threshold_list):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for i, race in enumerate(race_list):
        axs[i].plot(threshold_list, fmr_dict[race],
                    label='False Positive Rate (FPR)')
        axs[i].plot(threshold_list, fnmr_dict[race],
                    label='False Negative Rate (FNR)')
        axs[i].set_xlabel('Threshold')
        axs[i].set_ylabel('Rate')
        axs[i].set_title(f'{model} - {race}')
        axs[i].legend()
        axs[i].grid(True)
        axs[i].text(optimal_threshold_dict[race] + .15, 0.7,
                    s=f"Optimal Threshold: {optimal_threshold_dict[race]}", color='green')

    plt.tight_layout()
    plt.show()


def calculate_tp_tn_fp_fn(matching_pair_list, non_matching_pair_list, optimal_threshold):
    tp = sum(1 for distance in matching_pair_list if distance < optimal_threshold)
    tn = sum(1 for distance in non_matching_pair_list if distance >
             optimal_threshold)
    fp = len(matching_pair_list) - tp
    fn = len(non_matching_pair_list) - tn

    return tp, tn, fp, fn


def display_confusion_matrix(ax, matrix, accuracy):
    labels = ['Positive', 'Negative', 'Positive', 'Negative']
    values = np.array(matrix).reshape(2, 2)
    ax.matshow(values, cmap='Greens')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{values[i, j]}', ha='center', va='center')
    ax.set_xticklabels([''] + labels[:2])
    ax.set_yticklabels([''] + labels[2:])
    ax.set_xlabel('Actual', weight='bold')
    ax.set_ylabel('Predicted', weight='bold')
    ax.xaxis.set_label_position('top')

    ax.text(0.5, -0.1, f'Accuracy: {accuracy:.2f}', ha='center',
            va='center', transform=ax.transAxes)


def plot_confusion_matrices(model, race_list, matrix_dict, accuracy_dict):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for i, race in enumerate(race_list):
        display_confusion_matrix(
            axs[i], matrix_dict[race], accuracy_dict[race])
        axs[i].set_title(f'{model} - {race}', weight='bold')

    plt.tight_layout()
    plt.show()


def plot_bar_chart(model_list, race_list, matrix_dict, accuracy_dict, optimal_threshold_dict):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, race in enumerate(race_list):
        accuracies = [accuracy_dict[model][race] for model in model_list]
        axs[i].bar(model_list, accuracies, color='skyblue')
        axs[i].set_title(f'Accuracy Comparison - {race}')
        axs[i].set_xlabel('Model')
        axs[i].set_ylabel('Accuracy')
        axs[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def plot_metrics(model, race_list, matrix_dict):
    metrics = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'Specificity']
    fig, axs = plt.subplots(len(metrics), figsize=(15, 20))

    for j, metric in enumerate(metrics):
        for k, race in enumerate(race_list):
            tp, fp, fn, tn = matrix_dict[race]
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            specificity = tn / (tn + fp) if tn + fp != 0 else 0
            if metric == 'F1 Score':
                f1_score = 2 * (precision * recall) / (precision +
                                                       recall) if precision + recall != 0 else 0
                axs[j].bar(race, f1_score, color='orange')
            elif metric == 'Accuracy':
                axs[j].bar(race, accuracy, color='skyblue')
            elif metric == 'Precision':
                axs[j].bar(race, precision, color='green')
            elif metric == 'Recall':
                axs[j].bar(race, recall, color='red')
            elif metric == 'Specificity':
                axs[j].bar(race, specificity, color='purple')
            axs[j].set_title(f'{model} - {metric}')
            axs[j].set_ylabel(metric)
            axs[j].set_ylim(0, 1)
            axs[j].set_xticks([])  # Remove x ticks

    plt.tight_layout()
    plt.show()


def main():
    model_list = ['DeepFace', 'ArcFace', 'Facenet', 'Facenet512']
    # model_list = ['DeepFace']
    race_list = ['African', 'Asian', 'Caucasian', 'Indian']

    for model in model_list:
        fmr_dict = {}
        fnmr_dict = {}
        matrix_dict = {}
        accuracy_dict = {}
        optimal_threshold_dict = {}

        for race in race_list:
            file_path = f'testing_results/verification/{model}/{race}_results.txt'
            threshold_list = [(i / 100) for i in range(101)]

            matching_pair_list, non_matching_pair_list = read_data(file_path)
            fmr_list, fnmr_list, optimal_threshold = calculate_rate_and_threshold(
                matching_pair_list, non_matching_pair_list, threshold_list)
            tp, tn, fp, fn = calculate_tp_tn_fp_fn(
                matching_pair_list, non_matching_pair_list, optimal_threshold)

            print(f"{model}, {race} Optimal threshold: {optimal_threshold}")
            print(f"{model}, {race} TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

            fmr_dict[race] = fmr_list
            fnmr_dict[race] = fnmr_list
            matrix_dict[race] = (tp, fp, fn, tn)
            accuracy_dict[race] = calculate_accuracy(matrix_dict[race])
            optimal_threshold_dict[race] = optimal_threshold

        plot_rates(model, race_list, fmr_dict, fnmr_dict,
                   optimal_threshold_dict, threshold_list)
        plot_confusion_matrices(model, race_list, matrix_dict, accuracy_dict)

        # plot_metrics(model, race_list, matrix_dict)
    # plot_bar_chart(model_list, race_list, matrix_dict, accuracy_dict, optimal_threshold_dict)


if __name__ == "__main__":
    main()
