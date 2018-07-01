

from math import sqrt
import re
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
from face_recognition.cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path = "", n_neighbors = None, knn_algo = 'ball_tree', verbose=False):
    
    X = []
    y = []
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                if verbose:
                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                continue
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(class_dir)


    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically as:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path != "":
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

def predict(X_img_path, knn_clf = None, model_save_path ="", DIST_THRESH = .5):
    """
    recognizes faces in given image, based on a trained knn classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_save_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param DIST_THRESH: (optional) distance threshold in knn classification. the larger it is, the more chance of misclassifying an unknown person to a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'N/A' will be passed.
    """

    if not isfile(X_img_path) or splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_save_path == "":
        raise Exception("must supply knn classifier either thourgh knn_clf or model_save_path")

    if knn_clf is None:
        with open(model_save_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_img = face_recognition.load_image_file(X_img_path)
    X_faces_loc = face_locations(X_img)
    if len(X_faces_loc) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)


    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]

    # predict classes and cull classifications that are not with high confidence
    return [(pred, loc) if rec else ("N/A", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]

def draw_preds(img_path, preds):
    """
    shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param preds: results of the predict function
    :return:
    """
    source_img = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    for pred in preds:
        loc = pred[1]
        name = pred[0]
        # (top, right, bottom, left) => (left,top,right,bottom)
        draw.rectangle(((loc[3], loc[0]), (loc[1],loc[2])), outline="blue")
        draw.text((loc[3], loc[0] - 30), name, font=ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30))
    source_img.show()

if __name__ == "__main__":
    #knn_clf = train("knn_examples/train")
    cptCorrect =0
    cpt=0
    cptFalse=0
    cptUnkown=0
    for img_path in listdir("knn_examples/test"):
        print(img_path)
        preds = predict(join("knn_examples/test", img_path),knn_clf=None,model_save_path ="model/tarining_image_model")
        print(preds)
        #print(preds[cpt][0])
        draw_preds(join("knn_examples/test", img_path), preds)
        if (preds[0][0]!="N/A"):
            nameList=img_path.split(".")
            name=nameList[0]
            if(name==preds[0][0]):
                cptCorrect=cptCorrect+1
            else:

                cptFalse=cptFalse+1
        else:
            cptUnkown =cptUnkown+1
        cpt = cpt+1
    print("                   Resultats:               \n")
    print("nombre de tests : "+str(cpt))
    print("Taux de succes : " + str((cptCorrect/cpt)*100) +" %")
    print("Taux de faux positifs : " + str((cptFalse/cpt)*100) + " %")
    print("Taux de rejets : " + str((cptUnkown / cpt) * 100) + " %")


