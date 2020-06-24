# Handwrite_Text_Recognition
Recognises the handwriting text word in an image

# Dataset
The IAM On-Line Handwriting Database dataset is used for training the model. The dataset can be downloaded by the following link
https://drive.google.com/file/d/1PZAxlyQYc7Qp18UlHpZVifGnFR2n66fk/view?usp=sharing

# Requirements
Install the requirements for this project by simply 
sudo pip3 install -r requirements.txt

# Training

Download the dataset from the above link and keep words folder and words.txt file in data
For training go into the src folder and run the train.py file
python3 train.py
The model which is provided into repo is trainned till 27 epoches and it is saved 12.348896% validation character error rate.
So Word accuracy: 71.236111%. and Character error rate: 12.502647%
Colab training file Handwrite_Text_Recognition_Training.ipynb is given in the src folder for training in google colab.

# Validation 

For validating the training data run validation.py file in src floder.
python3 validation.py

The validation for this model is 
Character error rate: 12.348896%. Word accuracy: 71.750000%. as said in training process

# Types of tensorflow Decoders used
1. BestPath
2. BeamSearch

# Inference
For running on an text word image run

python3 main.py

It will run automatically on test-5.jpg images which is provided in data/test_images
Wanted to change the image give the image path in 18th line i.e, fnInfer = 'image path'

Want to run on beamSearch decoder run 

python3 main.py --beamsearch


