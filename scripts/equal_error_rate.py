import matplotlib.pyplot as plt

def read_data(file_path):
    positive_pairs = []
    negative_pairs = []

    with open(file_path, 'r') as file:
        file.readline()  # Skip the header
        count = 0
        for line in file:
            parts = line.split()
            distance = float(parts[7][:7])  # Extracting and converting the distance to float
            if parts[0] == parts[2]:
                positive_pairs.append(distance)
            else:
                negative_pairs.append(distance)

            count += 1
            if count > 10000:
                break

    return positive_pairs, negative_pairs

def calculate_rates(positive_pairs, negative_pairs, thresholds):
    fmr_list = []
    fnmr_list = []
    with open('tmp/temporary.txt', 'w') as file:
        pass

    min_value = 1
    for threshold in thresholds:
        false_positive = sum(1 for distance in positive_pairs if distance > threshold)
        true_negative = len(positive_pairs) - false_positive

        false_negative = sum(1 for distance in negative_pairs if distance < threshold)
        true_positive = len(negative_pairs) - false_negative

        fmr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) != 0 else 0
        fnmr = false_negative / (false_negative + true_positive) if (true_positive + false_negative) != 0 else 0

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


def plot_rates(model, race, fpr_list, fnr_list, optimal_threshold, threshold_list):
    plt.plot(threshold_list, fpr_list, label='False Positive Rate (FPR)')
    plt.plot(threshold_list, fnr_list, label='False Negative Rate (FNR)')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('Equal Error Rate (EER)')
    plt.legend()
    plt.xticks([i / 10 for i in range(11)])  # Set x-axis ticks at increments of 0.1
    plt.grid(True)
    plt.text(optimal_threshold -.39, 0.5, label='Optimal Threshold', s=f"Optimal Threshold: {optimal_threshold}", color='green')
    plt.show()

def main():
    model_list = ['DeepFace', 'ArcFace', 'Facenet', 'Facenet512']
    race_list = ['African', 'Asian', 'Caucasian', 'Indian']

    fmr_dict = {}
    fnmr_dict = {}
    optimal_threshold_dict = {}

    for model in model_list:
        for race in race_list:
            file_path = f'testing_results/verification/{model}/{race}_results.txt'
            threshold_list = [(i / 100) for i in range(101)]

            matching_pair_list, non_matching_pair_list = read_data(file_path)
            fmr_list, fnmr_list, optimal_threshold = calculate_rates        (matching_pair_list, non_matching_pair_list, threshold_list)

            print(f"{model}, {race} Optimal threshold: {optimal_threshold}")

            fmr_dict[race]= fmr_list
            fnmr_dict[race]= fnmr_list
            optimal_threshold_dict[race] = optimal_threshold

        plot_rates(model, race, fmr_dict, fnmr_dict, optimal_threshold_dict, threshold_list)

if __name__ == "__main__":
    main()
