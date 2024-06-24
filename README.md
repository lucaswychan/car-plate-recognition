<div align="center">
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-F63939?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://en.wikipedia.org/wiki/Optical_character_recognition">
    <img src="https://img.shields.io/badge/OCR-6600CC?style=for-the-badge" alt="OCR">
  </a>
</div>

<hr/>

# Car Plate Recognition
![output](/Image/image.png)

## Notice
For the detailed implementation and logic, please visit [REPORT.md](REPORT.md)

## Abstract
The project focuses on developing an optical character recognition (OCR) model for car plate recognition. The workflow includes two stages: model training and transfer learning. In the model training stage, a balanced EMNIST dataset is used to train a ResNet50 model with data augmentation techniques. Transfer learning is then applied in the second stage, where a custom dataset resembling car plate characters is used to fine-tune the pre-trained model. The model parameters and settings are carefully chosen to optimize performance. The trained OCR model is capable of recognizing car plate characters with high accuracy. Preprocessing steps are applied to input images, and the model outputs recognized characters and confidence levels. The model can be easily reused and deployed for future tasks or inference scenarios.

<hr/>

## Dependencies
You can install all the packages via
```
pip install -r requirements.txt
```

<hr/>

## Instructions
For simple usage, you can just run
```
python3 main.py
```
The images are stored in `license_images`. The trained OCR model is reloaded to perform OCR on car plate images. Preprocessing steps, including adding a border, grayscale conversion, resizing, and normalization, are applied to the input image. The processed image is then fed into the OCR model, which outputs recognized characters and corresponding confidence levels. If the confidence level is below 0.5, it is considered a recognition failure. The final output is the recognized car plate result, including any recognition failures.
<hr/>

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<hr/>
