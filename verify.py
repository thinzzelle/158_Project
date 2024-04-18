from deepface import DeepFace
import time
import tensorflow as tf
from datetime import datetime
import multiprocessing
from functools import partial

exception_list = []
exception_write_to_file_count = 0


def _write_exceptions_to_file(model):
    filename = 'tmp2/' + model + '/exceptions.txt'
    with open(filename, 'w') as file:
        for exception in exception_list:
            file.write(str(exception) + '\n')
    

def _write_test_result_to_file(template_folder, template_image_index, test_folder, test_image_index, result, results_file):

    results_file.write(
        f'{template_folder}\t'
        f'{template_image_index}\t'
        f'{test_folder}\t'
        f'{test_image_index}\t'
        f'{result}\t')

def get_image_from_pair(race, pair):
    if len(pair) == 4:
        template_folder = pair[0]
        template_index = int(pair[1])
        template_image = '_000' + str(template_index) + '.jpg'
        template_image_path = 'rfw/test/data/' + race + '/' + template_folder + '/' + template_folder + template_image

        test_folder = pair[2]
        test_index = int(pair[3])
        test_image = '_000' + str(test_index) + '.jpg'
        test_image_path = 'rfw/test/data' + race + '/' + test_folder + '/' + test_folder + test_image

    if len(pair) == 3:
        template_folder = pair[0]
        template_index = int(pair[1])
        template_image = '_000' + str(template_index) + '.jpg'
        template_image_path = 'rfw/test' + race + '/' + template_folder + '/' + template_folder + template_image

        test_folder = pair[0]
        test_index = int(pair[2])
        test_image = '_000' + str(test_index) + '.jpg'
        test_image_path = 'rfw/test'+ race + '/' + test_folder + '/' + test_folder + test_image

    else:
        raise Exception("Error in get_pair()")
    return template_folder, template_index, template_image_path, test_folder, test_index, test_image_path
    


def _run_tests(race, model, detector, euclidean_distance, pair_list, test_limit):

    global exception_list

    # Open results file for writing
    with open(f'tmp2/{model}/{race}_results.txt', 'w') as results_file:

        # Write the header of results file
        results_file.write('File1\tFile2\tResult')

        # iterate through each image in the folder
        count = 0
        for pair in pair_list:
            template_folder, template_index, template_image_path, test_folder, test_index, test_image_path = get_image_from_pair(pair)
            
            print(f"\n{template_image_path}\t{test_image_path}")

            try:
                # run model
                start_time = time.time()  # start the timer
                result = DeepFace.verify(template_image_path, test_image_path, model, detector, euclidean_distance)
                end_time = time.time()
                test_time = end_time - start_time
                print(f"Test Time: {test_time}")
                tf.keras.backend.clear_session()

                _write_test_result_to_file(template_folder, template_index, test_folder, test_index, result, results_file)
            except Exception as e:
                print(str(e))
                exception_info = [str(e), race, count, template_folder, template_image_path, test_image_path]
                exception_list.append(exception_info)

            count += 1
            if count > test_limit:
                break

def _init_values(race):
    pairs_file_path = 'rfw/test/txts/' + race + '/' + race + '_pairs.txt'

    pair_list = []
    with open(pairs_file_path, 'r') as f:
        for line in f:
            data = line.split('\t')
            data[-1] = data[-1].strip('\n')
            data_tuple = tuple(data)

            # only accept pairs from different files
            if len(data) == 4:
                pair_list.append(data_tuple)

    return pair_list


def main():
    # race_list = ['African', 'Asian', 'Caucasian', 'Indian']
    race_list = ['African']
    model_list = ['Facenet512']
    euclidean_distance = 'euclidean'
    detector = 'mtcnn'
    test_limit = 10
    
    for model in model_list:
        for race in race_list:
            pair_list = _init_values(race)

            _run_tests(race, model, detector, euclidean_distance, pair_list, test_limit)

        print(f"Output file generated successfully for {model}.")
        _write_exceptions_to_file(model)


if __name__ == "__main__":
    main()

# models = [
#   "VGG-Face", 
#   "Facenet", 
#   "Facenet512", 
#   "OpenFace", 
#   "DeepFace", 
#   "DeepID", 
#   "ArcFace", _
#   "Dlib", 
#   "SFace",
#   "GhostFaceNet",
# ]

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
