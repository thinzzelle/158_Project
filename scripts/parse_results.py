def update_metrics_from_file(filename, metrics):
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            folder, temp, test, match, prediction, result = parts[0].split(': ')[1], parts[1].split(': ')[1], parts[2].split(': ')[1], parts[3].split(': ')[1], parts[4].split(': ')[1], parts[5].split(': ')[1]

            # Update metrics based on result
            if result == 'tp':
                metrics['True Positive'] += 1
            elif result == 'tn':
                metrics['True Negative'] += 1
            elif result == 'fp':
                metrics['False Positive'] += 1
            elif result == 'fn':
                metrics['False Negative'] += 1

            # Update positive/negative test counts based on match
            if match == '0':
                metrics['Negative Test Count'] += 1
            elif match == '1':
                metrics['Positive Test Count'] += 1

def main():
    metrics = {'True Positive': 0, 'True Negative': 0, 'False Positive': 0, 'False Negative': 0, 'Positive Test Count': 0, 'Negative Test Count': 0}
    filename = 'African_results.txt'
    update_metrics_from_file(filename, metrics)
    print(metrics)

if __name__ == "__main__":
    main()
