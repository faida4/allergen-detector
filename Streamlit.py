import streamlit as st
import spacy
import pandas as pd
from PIL import Image
import pytesseract
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt

# Configuration de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#  Chargement du modèle
@st.cache_resource
def load_model(model_name):
    return spacy.load(model_name)


@st.cache_resource
def load_model():
    return spacy.load("model-allergenes")

nlp = load_model()


#  Fonction extraction format 1 : plat + prix sur la même ligne
def extract_menu_format1(image):
    image = image.convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    raw_text = pytesseract.image_to_string(binary, lang="fra")
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    
    plats = []
    i = 0
    while i < len(lines):
        match = re.match(r"^(.+?)\s+(\d{2,3})$", lines[i])
        if match:
            plat = match.group(1).strip(" .:-*")
            description = ""
            i += 1
            while i < len(lines) and not re.match(r"^(.+?)\s+(\d{2,3})$", lines[i]):
                description += " " + lines[i]
                i += 1
            plats.append({"Plat": plat, "Description": description.strip()})
        else:
            i += 1
    return pd.DataFrame(plats)

#  Fonction extraction format 2 : plat seul sur une ligne, description dessous
def extract_menu_format2(text_ocr):
    lines = [line.strip() for line in text_ocr.strip().split("\n") if line.strip()]

    plats = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Ignorer les lignes contenant uniquement des chiffres
        if re.match(r"^\d{1,3}$", line):
            i += 1
            continue

        # Heuristique : une ligne est un titre si elle commence par une majuscule,
        # ne contient pas de virgule, et fait moins de 5-6 mots
        if (len(line.split()) <= 6 and ',' not in line 
            and line[0].isupper() and not re.search(r"\d{2,3}$", line)):
            plat = line
            description = ""
            i += 1

            # Regrouper les lignes suivantes jusqu'à une nouvelle entrée probable
            while i < len(lines):
                next_line = lines[i]
                # Si ligne trop courte ou commence par majuscule et pas de virgule, considérer que c'est un nouveau plat
                if (
                    len(next_line.split()) <= 5
                    and ',' not in next_line
                    and next_line[0].isupper()
                    and not re.search(r"\d{2,3}$", next_line)
                ):
                    break
                if not re.match(r"^\d+$", next_line):  # ignorer les lignes numériques seules
                    description += " " + next_line
                i += 1

            plats.append({
                "Plat": plat.strip(),
                "Description": description.strip()
            })
        else:
            i += 1

    return pd.DataFrame(plats)


def detect_menu_format(lines):
    """
    Détecte automatiquement le format du menu :
    - Format 1 = plats avec prix (ex : "Pizza Margherita 19")
    - Format 2 = plats en titre seul, description en dessous
    """
    format1_matches = 0
    format2_candidates = 0

    for line in lines:
        # Format 1 : contient un nom + prix à la fin
        if re.match(r"^(.+?)\s+(\d{2,3})$", line):
            format1_matches += 1

        # Format 2 : lignes courtes, sans virgule ni chiffres
        elif len(line.split()) <= 6 and not re.search(r"[0-9]", line) and ',' not in line:
            format2_candidates += 1

    if format1_matches >= format2_candidates:
        return "format1"
    else:
        return "format2"

from unidecode import unidecode

def nettoyer_texte(texte):
    return unidecode(texte.lower())


#  Interface principale
st.markdown('<div class="big-title">🍽️ Détection d’allergènes dans un menu</div>', unsafe_allow_html=True)

# Charger le fichier CSS
def inject_css_from_file(file_path):
    with open(file_path) as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

# Utilisation
inject_css_from_file("style.css")

option = st.radio("Type de menu à analyser :", ["📸 Image (menu en photo)", "📝 Texte brut"])

df_menu = None
texte = ""

if option == "📸 Image (menu en photo)":
    uploaded_file = st.file_uploader("Téléversez une image de menu", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        st.image(image, caption="Image du menu", use_container_width=True)


        # OCR une seule fois
        image_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw_text = pytesseract.image_to_string(binary, lang="fra")
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        #  Détection automatique du format
        format_detected = detect_menu_format(lines)
        st.info(f" Format détecté automatiquement : {'Format 1 (Plat + prix)' if format_detected == 'format1' else 'Format 2 (Plat + description)'}")

        #  Extraction automatique
        if format_detected == "format1":
            df_menu = extract_menu_format1(image)
        else:
            df_menu = extract_menu_format2(raw_text)


elif option == "📝 Texte brut":
    texte = st.text_area("Collez ici le contenu du menu :", height=300)
    

    lignes = [ligne.strip() for ligne in texte.split("\n") if ligne.strip()]
    df_menu = pd.DataFrame({"Plat": lignes, "Description": lignes})

# Analyse avec SpaCy
if df_menu is not None and not df_menu.empty:
    resultats = []
    for _, row in df_menu.iterrows():
        
        texte_complet = f"{row['Plat']}. {row['Description']}"
        doc = nlp(texte_complet)

        allergenes = sorted(set(ent.label_ for ent in doc.ents))
        resultats.append({
            "Plat": row["Plat"],
            "Description": row["Description"],
            "Allergènes détectés": ", ".join(allergenes)
        })

    df_resultats = pd.DataFrame(resultats)
   
    st.markdown('<div class="section-header">📊 Résultats de la détection</div>', unsafe_allow_html=True)


    st.dataframe(df_resultats)

    # Statistiques
    all_allergenes = sum([r["Allergènes détectés"].split(", ") for r in resultats if r["Allergènes détectés"]], [])
    allergene_count = pd.Series(all_allergenes).value_counts()

    if not allergene_count.empty:
       
        st.markdown('<div class="section-header">📈 Fréquence des allergènes</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        allergene_count.plot(kind="bar", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    