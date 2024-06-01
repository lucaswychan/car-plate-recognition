import cv2
import torch
from PIL import Image
from torchvision import transforms

from ocr.resnet import ResNet


# Define the ResNet model
def get_model(num_classes=47):
    ocr_model = ResNet(num_classes=num_classes)
    ocr_model.load_state_dict(
        torch.load(
            "model/finetuned_ocr.pth",
            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
    )
    ocr_model.eval()

    return ocr_model


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (
                    0.1307,
                    0.1307,
                    0.1307,
                ),
                (
                    0.3081,
                    0.3081,
                    0.3081,
                ),
            ),
        ]
    )
    return transform(image).unsqueeze(0)


def enhance_image(image_path, border_size=3, border_color=255):
    # Add a border around the text region
    image = cv2.imread(image_path)
    image = cv2.copyMakeBorder(
        image,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )

    return image


def ocr(image_path):
    image = enhance_image(image_path)
    image = Image.fromarray(image)
    image = preprocess_image(image)

    ocr_model = get_model()

    with torch.no_grad():
        prediction = ocr_model(image)
        prediction = torch.softmax(prediction, dim=1)

    confidence, class_label = torch.max(prediction, 1)
    class_label = class_label.item()

    if 0 <= class_label <= 9:
        output = str(class_label)
    elif 10 <= class_label <= 35:
        output = chr(class_label + 55)
    else:
        output = chr(class_label + 61)

    return output, confidence
