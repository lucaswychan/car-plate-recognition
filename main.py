import os

from natsort import natsorted

from ocr_perform import ocr


def main():
    image_folder = "text_regions"
    car_plate = ""
    for file in natsorted(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, file)
        text, confidence = ocr(image_path=image_path)
        if confidence < 0.5:
            text = "_"
        car_plate += text

    print(car_plate)


if __name__ == "__main__":
    main()
