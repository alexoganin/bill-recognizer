from pathlib import Path

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from heic2png import HEIC2PNG
import matplotlib.pyplot as plt


def main():
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    # PDF
    # doc = DocumentFile.from_images(heic_to_png("data/raw/images/IMG_4765.HEIC"))
    doc = DocumentFile.from_pdf("data/raw/pdf/2024.02.27.pdf")
    # Analyze
    document = model(doc)
    document.show()
    print(document.pages)


def heic_to_png(filepath: str):
    path = Path(filepath)
    image = HEIC2PNG(filepath)
    image.save()
    return path.with_suffix(".png")


if __name__ == "__main__":
    main()