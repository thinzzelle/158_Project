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
    for threshold in thresholds:
        false_positive = sum(1 for distance in positive_pairs if distance > threshold)
        true_negative = len(positive_pairs) - false_positive

        false_negative = sum(1 for distance in negative_pairs if distance < threshold)
        true_positive = len(negative_pairs) - false_negative

        fmr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) != 0 else 0
        fnmr = false_negative / (false_negative + true_positive) if (true_positive + false_negative) != 0 else 0

        fmr_list.append(fmr)
        fnmr_list.append(fnmr)

        with open('tmp/temporary.txt', 'a') as file:
            file.write("Threshold: {:<10}\tFMR: {:.5f}\tFNMR: {:.5f}\tTP: {:<10}\tFP: {:<10}\tTN: {:<10}\tFN: {:<10}\n".format(threshold, round(fmr, 5), round(fnmr, 5), true_positive, false_positive, true_negative, false_negative))



    return fmr_list, fnmr_list

def plot_rates(fpr_list, fnr_list, threshold_list):
    plt.plot(threshold_list, fpr_list, label='False Positive Rate (FPR)')
    plt.plot(threshold_list, fnr_list, label='False Negative Rate (FNR)')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('Equal Error Rate (EER)')
    plt.legend()
    plt.xticks([i / 10 for i in range(11)])  # Set x-axis ticks at increments of 0.1
    plt.grid(True)
    plt.show()

def main():
    file_path = 'testing_results/verification/DeepFace/African_results.txt'
    threshold_list = [(i / 100) for i in range(101)]

    positive_pair_list, negative_pair_list = read_data(file_path)
    fmr_list, fnmr_list = calculate_rates(positive_pair_list, negative_pair_list, threshold_list)

    plot_rates(fmr_list, fnmr_list, threshold_list)

if __name__ == "__main__":
    main()
