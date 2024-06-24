from ocr.ocr_perform import ocr


def main():
    image_path = "license_images/GV5679.png"
    confidences, car_plate = ocr(image_path)
    print(f"Recognized license plate: {car_plate}")
    print(f"Confidence levels: {confidences}")


if __name__ == "__main__":
    main()
