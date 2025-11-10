import cv2
from scripts.licence_plate_detection import detect_license_plate
from scripts.character_recognition import character_segmentation, predict_characters

def license_plate_recognition(frame):
    license_plate = detect_license_plate(frame)
    if license_plate:
        for contour in license_plate:
            x, y, w, h = cv2.boundingRect(contour)
            license_plate_image = frame[y:y+h, x:x+w]
            annotated_characters, segmented_characters = character_segmentation(license_plate_image)
            characters = predict_characters(segmented_characters)
            license_plate_text = ""
            for character in characters:
                license_plate_text += character            
            cv2.putText(frame, license_plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    cap.set(2, 640)
    cap.set(3, 480)

    cap.set(cv2.CAP_PROP_FPS, 10)

    if not cap.isOpened():
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_license_plate = license_plate_recognition(frame)
        cv2.imshow("License Plate Recognition", frame_with_license_plate)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()