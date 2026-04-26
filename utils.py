import numpy as np
from PIL import Image

def preprocess_ecg(image):

    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def classify_risk(score):

    if score < 30:
        return "Low"
    elif score < 70:
        return "Moderate"
    else:
        return "High"


def calculate_final_risk(clinical_risk, ecg_risk):

    if ecg_risk is None:
        return clinical_risk

    return (clinical_risk * 0.6) + (ecg_risk * 0.4)