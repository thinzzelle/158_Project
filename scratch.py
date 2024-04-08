from deepface import DeepFace

# works with SFace, OpenFace, DeepID, Facenet and Facenet512
result = DeepFace.verify(img1_path='img1.jpg', img2_path='img2.jpg', model_name='OpenFace')          #returns boolean if first pic matches second pic
obj = DeepFace.analyze(img_path='img3.jpg', actions=['age', 'emotion', 'gender', 'race'])           #returns attribute to be printed out
found = DeepFace.find(img_path='img3.jpg', db_path='Jessica_and_Angelina/', model_name='Facenet')                         #returns location of second pic(s) matching the first pic

#uncomment below to just print the location of found images
#print(found[0]['identity'])
#uncomment below and will print image location and distance
#print(found)

print(result['verified'], result['model'], "Age:", obj[0]['age'], 'Emotion:', obj[0]['dominant_emotion'], 'Race:', obj[0]['dominant_race'], 'Gender:', obj[0]['dominant_gender'])


# Code Snippets

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