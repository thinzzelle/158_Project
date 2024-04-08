from deepface import DeepFace

exception_list = []
exception_verification_error_count = 0
exception_write_to_file_count = 0


def _write_exceptions_to_file():
    filename = 'exceptions.txt'
    with open(filename, 'w') as file:
        for exception in exception_list:
            file.write(str(exception) + '\n')
    print("Exceptions written to", filename)


def _print_results(races, metrics):
    for race in races:
        print("\nRace:", race)
        race_metrics = metrics[race]
        for key, value in race_metrics.items():
            print(key + ':', value)
        print("Total Tests:", race_metrics['Positive Test Count'] + race_metrics['Negative Test Count'], '\n')
    _write_exceptions_to_file()


def _get_template_image(race, folder):
    return 'rfw/test/data/' + race + '/' + folder + '/' + folder + '_0001.jpg', 1


def _get_test_image(race, folder, i):
    file = '_000' + str(i) + '.jpg'
    return 'rfw/test/data/' + race + '/' + folder + '/' + folder + file, i


# returns true if template image and test image are the same person
def _test_for_match(pair_list, template_image_index, test_image_index):
    for pair in pair_list:
        if pair[0] == template_image_index and pair[1] == test_image_index:
            return True
        elif pair[0] == test_image_index and pair[1] == template_image_index:
            return True
    return False


def _calculate_results(is_match, result, race_metrics):
    if is_match:
        race_metrics['Positive Test Count'] += 1
    else:
        race_metrics['Negative Test Count'] += 1

    print("Is Match?:       ", is_match)
    print("Model Prediction:", result['verified'])

    if is_match and result['verified'] == True:
        print("Result: \t  True Positive")
        race_metrics["True Positive"] += 1
    elif is_match and not result['verified']:
        print("Result: \t  False Negative")
        race_metrics["False Negative"] += 1
    elif not is_match and result['verified']:
        print("Result: \t  False Positive")
        race_metrics["False Positive"] += 1
    elif not is_match and not result['verified']:
        print("Result: \t  True Negative")
        race_metrics["True Negative"] += 1


def _write_test_result_to_file(template_image_index, test_image_index, is_match, result, folder, results_file):
    results_file.write(
        f'folder: {folder}'
        f'\ttemp: {template_image_index}'
        f'\t test: {test_image_index}'
        f'\t match?: {int(is_match)}'
        f'\tprediction: {int(result["verified"])}'
        f'\tResult: ')

    if is_match and result['verified']:
        results_file.write("tp\n")
    elif is_match and not result['verified']:
        results_file.write("fn\n")
    elif not is_match and result['verified']:
        results_file.write("fp\n")
    elif not is_match and not result['verified']:
        results_file.write("tn\n")


def _run_tests(race, model, detector, folder_size_list, lookup_table, metrics):
    print("\n||||||||||||||||||||||||||| Race:", race, "|||||||||||||||||||||||||||||||||||||||||||||")

    global exeption_list
    global exception_write_to_file_count
    global exception_verification_error_count

    # Open results file for writing
    with open(f'{race}_results.txt', 'w') as results_file:

        # iterate through each folder
        count = 0
        for folder, size in folder_size_list:
            template_image, template_image_index = _get_template_image(race, folder)
            print("\n--------------- Testing Folder:", folder, "------------------")

            # run tests
            for i in range(2, size + 1):
                test_image, test_image_index = _get_test_image(race, folder, i)
                print("\nTemplate:", template_image, "\tTest:", test_image)
                pair_list = lookup_table[folder]

                # run model and store result
                try:
                    result = DeepFace.verify(template_image, test_image, model, detector)

                    # is the template and test the same person?
                    is_match = _test_for_match(pair_list, template_image_index, test_image_index)
                    _calculate_results(is_match, result, metrics[race])

                except Exception as e:
                    print("An exception occured:", str(e))
                    exception_list.append(e)
                    exception_verification_error_count += 1

                # write results of test to file
                try:
                    _write_test_result_to_file(template_image_index, test_image_index, is_match, result, folder, results_file)
                except Exception as e:
                    print("An exception occurred while writing results to file:", str(e))
                    exception_list.append(e)
                    exception_write_to_file_count += 1

            if count > 1:
                break
            count += 1


def _init_values(race):
    pairs_file_path = 'rfw/test/txts/' + race + '/' + race + '_pairs.txt'
    people_file_path = 'rfw/test/txts/' + race + '/' + race + '_people.txt'

    folder_size = []
    with open(people_file_path, 'r') as f:
        for line in f:
            group, num_people = line.strip().split('\t')
            folder_size.append((group, int(num_people)))

    lookup_table = {}
    with open(pairs_file_path, 'r') as f:
        for line in f:
            data = line.strip().split('\t')

            # only accept pairs from the same file
            if len(data) == 3:
                key, value = data[0], [int(data[1]), int(data[2])]
                if key in lookup_table:
                    lookup_table[key].append(value)
                else:
                    lookup_table[key] = [value]

    return folder_size, lookup_table


def _init_metrics(race, metrics):
    metrics[race] = {
        'True Positive': 0,
        'True Negative': 0,
        'False Positive': 0,
        'False Negative': 0,
        'Positive Test Count': 0,
        'Negative Test Count': 0
    }


# Main function
def main():
    # races = ['African', 'Asian', 'Caucasian', 'Indian']
    races = ['African']

    # DeepFace settings
    model = 'Facenet'
    detector = 'mtcnn'

    # stores true positive, true negative, false positive, false negative, positive test count, negative test count for each race
    metrics = {}

    for race in races:
        folder_size_list = []  # list of tuples. each tuple contains the folder name and number of people in folder
        lookup_table = {}  # a dictionary to stores pairs from the same folder only.
        folder_size_list, lookup_table = _init_values(race)
        _init_metrics(race, metrics)

        _run_tests(race, model, detector, folder_size_list, lookup_table, metrics)

    _print_results(races, metrics)


if __name__ == "__main__":
    main()
