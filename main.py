from deepface import DeepFace
import pandas as pd
import os

# Load the FairFace dataset
fairface_data = pd.read_csv('fair_face/archive/fairface_label_val.csv')
image_dir = 'C:/Users/quris/Documents/fair_face/pythonProject2/fair_face/img'
age_start = 0
age_end = 0
images = 300

# Function to evaluate predictions
def evaluate_predictions(predictions, ground_truth):
    #Uncomment to see what the age range is
    #print(age_ground_truth)
    #Uncomment to see what the age prediction is
    #print(predictions)
    correct = 0
    total = len(predictions)
    for pred, gt in zip(predictions, ground_truth):
        #print('gt is:', gt)
        age_range_split = gt.split('-')
        age_range_start = int(age_range_split[0])
        age_range_end = int(age_range_split[1])
        if age_range_start <= pred <= age_range_end:
            correct += 1
    accuracy = (correct / total) * 100
    return accuracy
def evaluate_other_predictions(predictions, ground_truth):
    #print('predictions', predictions)
    #print('ground truth', ground_truth)
    other_correct = 0
    other_total = len(predictions)
    for pred, gt in zip(predictions, ground_truth):
        if pred == gt:
            other_correct += 1
    other_accuracy = (other_correct / other_total) * 100
    return other_accuracy

# Function to predict age, gender, and race
def predict_age_gender_race(image_path):
    obj = DeepFace.analyze(image_path, enforce_detection=False)
    age = obj[0]['age']
    gender = obj[0]['dominant_gender']
    race = obj[0]['dominant_race']
    return age, gender, race

# Test predictions on the FairFace dataset
age_predictions = []
gender_predictions = []
race_predictions = []
age_ground_truth = fairface_data['age']
gender_ground_truth = fairface_data['gender']
race_ground_truth = fairface_data['race']

for idx, row in fairface_data.iterrows():
    image_path = os.path.join(image_dir, row['file'])
    #print("Image path", image_path)
    age_pred, gender_pred, race_pred = predict_age_gender_race(image_path)
    age_predictions.append(age_pred)
    gender_predictions.append(gender_pred)
    race_predictions.append(race_pred)

# Evaluate accuracy
age_accuracy = evaluate_predictions(age_predictions, age_ground_truth)
gender_accuracy = evaluate_other_predictions(gender_predictions, gender_ground_truth)
race_accuracy = evaluate_other_predictions(race_predictions, race_ground_truth)

with open('results.txt', 'w') as file:
    file.write("Accuracy for predicting age: {:.2f}%\n".format(age_accuracy))
    file.write("Accuracy for predicting gender: {:.2f}%\n".format(gender_accuracy))
    file.write("Accuracy for predicting race: {:.2f}%\n".format(race_accuracy))
print("Accuracy for predicting age: {:.2f}%".format(age_accuracy))
print("Accuracy for predicting gender: {:.2f}%".format(gender_accuracy))
print("Accuracy for predicting race: {:.2f}%".format(race_accuracy))
