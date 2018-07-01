
This is a project of using the k-nearest-neighbors(knn) algorithm for face recognition.

This project is useful when you whish to recognize a large set of known people, and make a prediction for an unkown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled(known) faces, and can then predict the person in an unkown image by finding the k most similar faces(images with closet face-features under eucledian distance) in its training set, and performing a majority vote(possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden and two images of Obama, The result would be 'Obama'.
*This implemententation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Organisation :
-First, prepare a set of images of the known people you want to recognize.
 Organize the images in a single directory with a sub-directory for each known person.
-Then, call the 'train' function with the appropriate parameters.
 make sure to pass in the 'model_save_path' if you want to re-use the model without having to re-train it. 
-After training the model, you can call 'predict' to recognize the person in an unknown image.

ps : these steps are done. you can do them again if you to retrain the model. otherwise you can just execute the script 

NOTE: This example requires scikit-learn to be installed! You can install it with pip:
$ pip3 install scikit-learn
