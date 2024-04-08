# from deepface import DeepFace
#
# backends = [
#   'opencv',
#   'ssd',
#   'dlib',
#   'mtcnn',
#   'retinaface',
#   'mediapipe',
#   'yolov8',
#   'yunet',
#   'fastmtcnn',
# ]
# detector_backend = backends[1]
#
#
# img1_path = 'rfw/test/data/African/m.0b0pdf/m.0b0pdf_0002.jpg'
# # img2_path = 'rfw/test/data/African/m.0b0pdf/m.0b0pdf_0003.jpg'
#
# img2_path = 'rfw/test/data/African/m.0fpgg4/m.0fpgg4_0003.jpg'
#
#
# model_name = 'Facenet'
#
#
# # works with SFace, OpenFace, DeepID, Facenet and Facenet512
# result = DeepFace.verify(img1_path, img2_path, model_name, detector_backend)          #returns boolean if first pic
#
# # print(result['verified'], result['model'], "Age:", obj[0]['age'], 'Emotion:', obj[0]['dominant_emotion'], 'Race:', obj[0]['dominant_race'], 'Gender:', obj[0]['dominant_gender'])
# print(result['verified'], result['model'])


from deepface import DeepFace

# Function to count true positives, true negatives, false positives, and false negatives

def _print_results(races, metrics):
    for race in races:
        print("\nRace:", race)
        race_metrics = metrics[race]
        for key, value in race_metrics.items():
            print(key + ':', value)
        print("Total Tests:", race_metrics['Positive Test Count'] + race_metrics['Negative Test Count'], '\n')

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

def _calculate_results(is_match, result, metrics):
    if is_match == True:
        metrics['Positive Test Count'] +=1
    else:
        metrics['Negative Test Count'] += 1

    print("is_match", is_match)
    print("result", result['verified'])

    if is_match == True and result['verified'] == True:
        print("Result True Positive")
        metrics["True Positive"] += 1
    elif is_match == True and result['verified'] == False:
        print("Result False Negative")
        metrics["False Negative"] += 1
    elif is_match == False and result['verified'] == True:
        print("Result False Positive")
        metrics["False Positive"] += 1
    elif is_match == False and result['verified'] == False:
        print("Result True Negative")
        metrics["True Negative"] += 1

def _run_tests(race, model, detector, folder_size_list, lookup_table, metrics):
    print("\nRace:", race, "||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

    # iterate through each folder
    count = 0
    for folder, size in folder_size_list:
        template_image, template_image_index = _get_template_image(race, folder)
        print("\n------ Testing Folder:", folder, "--------------------------------------------------")

        # run tests
        for i in range(2, size+1):
            test_image, test_image_index = _get_test_image(race, folder, i)
            print("\n\tTemplate:", template_image, "\tTest:", test_image)
            pair_list = lookup_table[folder]

            # run the model and store the result
            result = DeepFace.verify(template_image, test_image, model, detector)

            # is the template and test the same person?
            is_match = _test_for_match(pair_list, template_image_index, test_image_index)
            _calculate_results(is_match, result, metrics)

        if count > 1:
            break
        count +=1

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
    races = ['African', 'Asian', 'Caucasian', 'Indian']
    # races = ['African']

    # DeepFace settings
    model = 'Facenet'
    detector = 'mtcnn'

    # stores true positive, true negative, false positive, false negative, positive test count, negative test count for each race
    metrics = {}

    for race in races:
        folder_size_list = []   # list of tuples. each tuple contains the folder name and number of people in folder
        lookup_table = {}       # a dictionary to stores pairs from the same folder only.
        folder_size_list, lookup_table = _init_values(race)
        _init_metrics(race, metrics)

        _run_tests(race, model, detector, folder_size_list, lookup_table, metrics[race])

    _print_results(races, metrics)


if __name__ == "__main__":
    main()