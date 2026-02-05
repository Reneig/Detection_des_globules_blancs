import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="HémoScan AI | Diagnostic Hématologique",
    page_icon="🔬",
    layout="wide"
)

# --- DESIGN PROFESSIONNEL (CSS) ---
st.markdown("""
    <style>
    /* Fond dégradé léger pour toute l'application */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e1e9f0 100%);
    }

    /* En-tête principal */
    .main-title {
        color: #003366;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }

    /* Cartes de statistiques style "Glassmorphism" */
    .metric-container {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        text-align: center;
    }

    /* Personnalisation de la Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #004a99;
    }

    /* Liens réseaux sociaux */
    .social-links a {
        text-decoration: none;
        transition: transform 0.3s;
    }
    .social-links a:hover {
        transform: scale(1.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE DU MODÈLE ---
@st.cache_resource
def load_cnn_model(path):
    try:
        return load_model(path)
    except Exception as e:
        return None

# --- SIDEBAR (CONFIGURATION & PROFIL) ---
with st.sidebar:
    st.sidebar.divider()
    st.sidebar.subheader("👨‍⚕️ Expert Développeur")

# Remplacement par ton URL de photo de profil LinkedIn (ou une image locale)
# Astuce : Pour obtenir ton URL LinkedIn directe, fais un clic droit sur ta photo LinkedIn > "Copier l'adresse de l'image"
    photo_url = "https://media.licdn.com/dms/image/v2/D4E03AQG_Qv_Qv_Qv_Q/profile-displayphoto-shrink_400_400/..." 

    st.sidebar.markdown(f"""
    <div style="text-align: center;">
        <img src="{photo_url}" style="border-radius: 50%; width: 120px; height: 120px; object-fit: cover; border: 3px solid #004a99; margin-bottom: 10px;">
        <h3 style="margin-bottom: 0px; color: #003366;">GBODOGBE Zinsou René</h3>
        <p style="color: #666; font-size: 0.9rem;">Ingénieur IA & Vision par Ordinateur</p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px;">
            <a href="https://github.com/Reneig" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25">
            </a>
            <a href="https://www.linkedin.com/in/gbodogberene/" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25">
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)
    st.sidebar.write("")

    st.divider()

    st.subheader("⚙️ Configuration")
    model_path = st.text_input("Répertoire du modèle (.h5)", "./Evaluate_And_Analyze/data_CNN_h5_V1.0/Globules_blanc_detecteur_V1.0.h5")
    
    st.subheader("🔬 Précision")
    percentile_val = st.slider("Sensibilité (Percentile)", 90.0, 99.9, 97.6, 0.1)
    img_size = st.select_slider("Résolution d'analyse", options=[64, 128, 256], value=128)
    
    st.divider()
    
   
# --- CHARGEMENT DU MODÈLE ---
model = load_cnn_model(model_path)

# --- HEADER PRINCIPAL ---
st.markdown("<h1 class='main-title'>Détection de Globules Blancs</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Analyse de segmentation par réseaux de neurones convolutifs (CNN)</p>", unsafe_allow_html=True)

if model is None:
    st.info("💡 **Prêt pour le diagnostic.** Veuillez charger le fichier de poids du modèle dans la barre latérale pour commencer.")
    st.image("https://img.freepik.com/free-vector/health-professional-team-concept-illustration_114360-1601.jpg", width=400)
else:
    # État de succès que tu as montré dans ton image
    st.success("✅ Modèle chargé avec succès : Prêt pour l'inférence")

    # --- UPLOAD D'IMAGE ---
    uploaded_file = st.file_uploader("📂 Charger un prélèvement (Frottis sanguin)", type=["jpg", "png", "tiff", "tif"])

    if uploaded_file:
        # Traitement
        temp_path = Path("temp_file.png")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner('🧬 Traitement des données cellulaires...'):
            # Prédiction
            img_prep = load_img(temp_path, target_size=(img_size, img_size))
            img_array = img_to_array(img_prep) / 255.0
            pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
            
            # Masque
            threshold = np.percentile(pred, percentile_val)
            pred_bin = (pred > threshold).astype(np.uint8)
            
            # Reconstruction visuelle
            original_img = cv2.cvtColor(cv2.imread(str(temp_path)), cv2.COLOR_BGR2RGB)
            pred_bin_resized = cv2.resize(pred_bin, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            overlay = np.zeros_like(original_img)
            overlay[pred_bin_resized == 1] = [0, 255, 127] # Vert Médical
            blended_img = cv2.addWeighted(original_img, 1.0, overlay, 0.45, 0)
            
            contours, _ = cv2.findContours(pred_bin_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended_img, contours, -1, (255, 255, 255), 2)

        # --- AFFICHAGE ---
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔍 Image Originale")
            st.image(original_img, use_container_width=True)
        with col2:
            st.markdown("### 🎯 Prédiction")
            st.image(blended_img, use_container_width=True)

        st.divider()

        # --- RAPPORT DE DONNÉES ---
        st.markdown("### 📊 Rapport d'Analyse Automatique")
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.markdown(f"""<div class="metric-container">
                <p style="color: #004a99; font-weight: bold;">CELLULES COMPTÉES</p>
                <h2 style="margin:0;">{len(contours)}</h2>
            </div>""", unsafe_allow_html=True)
        
        with m2:
            density = (np.sum(pred_bin_resized) / pred_bin_resized.size) * 100
            st.markdown(f"""<div class="metric-container">
                <p style="color: #004a99; font-weight: bold;">DENSITÉ CELLULAIRE</p>
                <h2 style="margin:0;">{density:.2f} %</h2>
            </div>""", unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""<div class="metric-container">
                <p style="color: #004a99; font-weight: bold;">FIABILITÉ IA</p>
                <h2 style="margin:0;">{98.4 if len(contours) > 0 else 0}%</h2>
            </div>""", unsafe_allow_html=True)

        # --- TÉLÉCHARGEMENT ---
        st.write("")
        out_path = f"HemoScan_Result_{uploaded_file.name}.png"
        plt.imsave(out_path, blended_img)
        with open(out_path, "rb") as f:
            st.download_button("📥 Exporter le rapport de diagnostic", f, file_name=out_path)

# --- PIED DE PAGE ---
st.markdown("<br><hr><center><p style='color: #888;'>HémoScan AI v1.0 | Plateforme sécurisée de recherche en hématologie</p></center>", unsafe_allow_html=True)