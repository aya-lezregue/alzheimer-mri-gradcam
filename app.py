from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models, transforms as T
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------- Configuration ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "resnet50_phase2_best.pth"
CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]  # ordre = entraînement
LABELS_FR = {
    "MildDemented": "Léger",
    "ModerateDemented": "Modéré",
    "NonDemented": "Sain (non dément)",
    "VeryMildDemented": "Très léger",
}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Mêmes transformations que pendant l'évaluation (étape 5)
eval_tf = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

def denormalize(t):
    t = t.clone()
    for c in range(3):
        t[c] = t[c] * STD[c] + MEAN[c]
    return t.clamp(0, 1)

# ---------- Modèle (chargé une seule fois) ----------
@st.cache_resource
def load_model():
    m = models.resnet50(weights=None)            # pas besoin d'ImageNet : on charge nos poids
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 4))
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    m.eval()
    for p in m.parameters():                      # Grad-CAM a besoin des gradients
        p.requires_grad_(True)
    return m

def explain(model, pil_img, target_class, method, heat_strength):
    x = eval_tf(pil_img.convert("L")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    pred = int(probs.argmax())
    cls = pred if target_class is None else target_class

    cam = method(model=model, target_layers=[model.layer4[-1]])
    heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]

    rgb = denormalize(x[0]).permute(1, 2, 0).numpy().astype(np.float32)
    overlay = show_cam_on_image(rgb, heat, use_rgb=True, image_weight=1 - heat_strength)
    return probs, pred, heat, overlay

# ---------- Interface ----------
st.set_page_config(page_title="Alzheimer : IRM et Grad-CAM", layout="wide")
st.title("Classification des stades d'Alzheimer par IRM")
st.warning("Outil expérimental à but pédagogique. Il ne constitue pas un diagnostic médical "
           "et n'a fait l'objet d'aucune validation clinique.")

with st.sidebar:
    st.header("Paramètres")
    method_name = st.selectbox("Méthode d'explication", ["Grad-CAM", "Grad-CAM++"])
    target_choice = st.selectbox("Classe à expliquer",
                                 ["Classe prédite"] + [LABELS_FR[c] for c in CLASSES])
    heat_strength = st.slider("Intensité de la carte de chaleur", 0.1, 0.9, 0.5, 0.1)
    st.markdown("---")
    st.caption("Modèle : ResNet50 pré-entraîné (ImageNet), affiné sur le dataset "
               "« Alzheimer's Dataset (4 class of Images) » (images 2D, niveaux de gris).")
    st.caption("Les résultats ne sont fiables que pour des images de même type que celles "
               "de l'entraînement (coupes axiales 2D, cerveau seul).")

if not MODEL_PATH.exists():
    st.error(f"Modèle introuvable : {MODEL_PATH}. "
             "Téléchargez resnet50_phase2_best.pth (lien dans le README) "
             "et placez-le dans le dossier models/.")
    st.stop()

model = load_model()

file = st.file_uploader("Importer une IRM (JPG ou PNG)", type=["jpg", "jpeg", "png"])

if file is None:
    st.info("Importez une image IRM pour lancer l'analyse.")
else:
    img = Image.open(file)
    method = GradCAM if method_name == "Grad-CAM" else GradCAMPlusPlus
    target = None if target_choice == "Classe prédite" else \
             [LABELS_FR[c] for c in CLASSES].index(target_choice)

    with st.spinner("Analyse en cours..."):
        probs, pred, heat, overlay = explain(model, img, target, method, heat_strength)

    top = CLASSES[pred]
    st.subheader(f"Prédiction : {LABELS_FR[top]} ({probs[pred]:.1%})")
    if probs[pred] < 0.60:
        st.info("Confiance faible : le modèle hésite entre plusieurs stades. Interprétez avec prudence.")

    c1, c2, c3 = st.columns(3)
    c1.image(img.convert("L").resize((224, 224)), caption="IRM importée", use_container_width=True)
    c2.image(heat, caption="Carte Grad-CAM (rouge = forte influence)", use_container_width=True,
             clamp=True)
    c3.image(overlay, caption="Superposition", use_container_width=True)

    st.markdown("#### Probabilités par classe")
    dfp = pd.DataFrame({"Classe": [LABELS_FR[c] for c in CLASSES], "Probabilité": probs}).set_index("Classe")
    st.bar_chart(dfp)
    st.dataframe(dfp.style.format("{:.1%}"), use_container_width=True)

    st.caption("La carte montre les zones qui ont influencé la décision du modèle, pas des lésions "
               "confirmées. Sa résolution est limitée (7×7 avant agrandissement).")