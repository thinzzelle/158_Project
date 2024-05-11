# Demographic Face Recognition

This repository contains the main program for testing pairs in the context of demographic facial recognition. The program evaluates the performance of various deep learning models such as FaceNet, Facenet512, DeepFace, and ArcFace among different racial demographics. 

## Usage

1. **Dataset Preparation**: The RFW dataset must be downloaded separately and stored in the `rfw/` folder. Additionally, ensure that the `Pairs.txt` file is present, which holds the pairs files for testing.

2. **Dependencies**: Ensure you have installed the required dependencies. This project utilizes TensorFlow and the `deepface` library for facial recognition. Install dependencies using `pip install -r requirements.txt`.

3. **Execution**: Execute the `main.py` script to run the tests. The script will iterate through the specified race categories (African, Asian, Caucasian, Indian) and evaluate the specified models on the provided dataset.

## Files

- `main.py`: Contains the main script for running the tests.
- `rfw/`: Directory for storing the RFW dataset.
- `tmp/`: Directory for storing temporary files and exceptions generated during testing.
- `requirements.txt`: List of dependencies.

## How it Works

1. The script initializes the specified models and parameters.
2. It reads pairs of images from the provided dataset.
3. For each pair, the script performs facial verification using the chosen model and detector.
4. Results are logged in a results file, and exceptions (if any) are recorded separately.
5. Finally, the output files are generated for each model, containing the test results.

## Note

- Ensure the correct file paths are configured for the dataset and pairs files.
- Adjust the `test_limit` variable to control the number of tests performed per race category.

## Models and Detectors

- **Models**: FaceNet, FaceNet512, DeepFace, ArcFace
- **Distance Metric**: Cosine
- **Detector**: MTCNN

## References

- [RFW Dataset](https://github.com/davidsonic/RFW)
- [DeepFace Library](https://github.com/serengil/deepface)

