import cv2
import numpy as np
import face_recognition
import os

def findEncodeings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def main():
    path = 'persons'
    images = []
    classNames = []
    personsList = os.listdir(path)

    for cl in personsList:
        curPersonn = cv2.imread(f'{path}/{cl}')
        images.append(curPersonn)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    encodeListKnown = findEncodeings(images)
    print('Encoding Complete.')

    # Load the image for face detection
    image_path = 'test_image.jpg'
    img = cv2.imread(image_path)
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurentFrame = face_recognition.face_locations(imgS)
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)

    for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        matchIndices = [i for i, match in enumerate(matches) if match]

        for matchIndex in matchIndices:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Face Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
