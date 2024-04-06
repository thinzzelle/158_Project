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
import os

# Function to count true positives, true negatives, false positives, and false negatives
def count_results(image_list, pairs_file, model_name, detector_backend, num_photos):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    test_count = 0

    with open(pairs_file, 'r') as f:
        for line in f:
            if test_count >= num_photos:  # Check a given number of photos
                break

            try:
                img1_id, img1_label, img2_id, img2_label = line.strip().split('\t')
                img1_path = [img_path for img_path, label in image_list if img1_id in img_path][0]
                img2_path = [img_path for img_path, label in image_list if img2_id in img_path][0]

                # Compare images
                result = DeepFace.verify(img1_path, img2_path, model_name, detector_backend)

                if result['verified']:
                    if img1_label == img2_label:
                        true_positives += 1
                    else:
                        false_positives += 1
                else:
                    if img1_label == img2_label:
                        false_negatives += 1
                    else:
                        true_negatives += 1

                test_count += 1

            except Exception as e:
                print("Exception occurred:", str(e))

    return true_positives, true_negatives, false_positives, false_negatives, test_count

# Main function
def main():
    # Folder and file paths
    data_folder = 'rfw/test/data'
    txt_folder = 'rfw/test/txts'
    races = ['African', 'Asian', 'Caucasian', 'Indian']
    num_photos = 3  # Number of photos to test with

    # DeepFace settings
    model_name = 'Facenet'
    detector_backend = 'mtcnn'

    for race in races:
        pairs_file = txt_folder + '/' + race + '/' + race + '_pairs.txt'
        images_file_path = txt_folder + '/' + race + '/' + race + '_images.txt'
        people_file_path = txt_folder + '/' + race + '/' + race + '_people.txt'

        print(pairs_file, images_file_path, people_file_path)

        continue

        # Read image list
        image_list = []
        with open(images_file_path, 'r') as f:

            # for line in f:
            for i in range(5):
                img_path, label = line.strip().split('\t')
                image_list.append((os.path.join(data_folder, race, img_path), label))

        # Read number of people
        with open(people_file_path, 'r') as f:
            num_people = int(f.readline().strip().split('\t')[1])

        print("Race:", race)
        true_positives, true_negatives, false_positives, false_negatives, test_count = count_results(
            image_list, pairs_file, model_name, detector_backend, num_photos
        )

        print("True Positives:", true_positives)
        print("True Negatives:", true_negatives)
        print("False Positives:", false_positives)
        print("False Negatives:", false_negatives)
        print("Total Tests:", test_count)
        print("Total People:", num_people)
        print()

if __name__ == "__main__":
    main()
