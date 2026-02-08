import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# CONFIGURATION DE LA PAGE STREAMLIT
st.set_page_config(
    page_title="Traffic counter",
    page_icon="🚦",
    layout="wide"
)

with st.sidebar:
    st.header("Configuration")
    
    st.subheader("1. Source Vidéo")
    video_file = st.file_uploader("Importer une vidéo MP4", type=['mp4', 'avi', 'mov'])
    with st.expander("Intérêt de l'étape"):
        st.write("""
        L'algorithme a besoin d'une matrice de pixels (la vidéo) pour travailler. 
        Streamlit charge le fichier en mémoire RAM, mais OpenCV a besoin d'un chemin physique. 
        On créera donc un fichier temporaire.
        """)

    st.subheader("2. Paramètres IA")
    conf_threshold = st.slider("Seuil de Confiance (Confidence)", 0.0, 1.0, 0.1)
    line_pos = st.slider("Position de la ligne (0=Haut, 1=Bas)", 0.0, 1.0, 0.6)
    
    with st.expander("Comprendre ces paramètres"):
        st.write(f"""
        **Seuil de confiance ({conf_threshold})** : 
        C'est le niveau d'exigence.L'objet est ignoré si l'IA est moins de{int(conf_threshold*100)}% sûre, pour éviter les faux positifs.
        
        **Ligne ({line_pos})** : 
        L'endroit où le comptage s'effectue.
        *Intérêt : Doit être placé là où la vue est la plus dégagée.*
        """)

    st.markdown("---")
    st.markdown("### Légende")
    st.markdown("🟦 **Voiture** | 🟥 **Moto**")
    st.markdown("🟨 **Bus** | 🟩 **Camion**")


# LOGIQUE DE COMPTAGE (CLASSES)
class TrafficCounter:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        
        # Dictionnaire : les numéros COCO correspondent aux types de véhicules
        # 2=Voiture, 3=Moto, 5=Bus, 7=Camion dans la base de données COCO
        self.target_classes = {2: 'Voiture', 3: 'Moto', 5: 'Bus', 7: 'Camion'}
        
        # Historique des positions de chaque véhicule (pour tracer leur trajet)
        self.vehicle_trails = {}
        
        # Liste des véhicules déjà comptés (pour éviter de compter 2 fois)
        self.counted_ids = set()
        
        # Compteurs : total et par type de véhicule
        self.total_count = 0
        self.counts_by_class = {'Voiture': 0, 'Moto': 0, 'Bus': 0, 'Camion': 0}

    def process_frame(self, frame, line_position_ratio, conf_thresh):
        # Récupération des dimensions de l'image
        height, width = frame.shape[:2]
        
        # Calcul de la position Y de la ligne (en pixels)
        # Exemple : si line_position_ratio=0.6, la ligne sera à 60% de la hauteur
        line_y = int(height * line_position_ratio)
        
        # YOLO détecte et suit les véhicules (persist=True garde les mêmes IDs)
        results = self.model.track(frame, persist=True, verbose=False, conf=conf_thresh)
        
        # Vérifier si des objets ont été détectés
        if results[0].boxes.id is not None:
            # Extraction des informations de détection
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Coordonnées des rectangles
            track_ids = results[0].boxes.id.int().cpu().tolist()  # IDs uniques
            classes = results[0].boxes.cls.int().cpu().tolist()  # Types d'objets

            # Boucle sur chaque véhicule détecté
            for box, track_id, cls in zip(boxes, track_ids, classes):
                # Ne traiter que les véhicules qui nous intéressent
                if cls in self.target_classes:
                    x1, y1, x2, y2 = box
                    
                    # Calcul du point de référence : centre en bas du véhicule
                    cx, cy = int((x1 + x2) / 2), int(y2)
                    vehicle_type = self.target_classes[cls]

                    # Créer l'historique si c'est la première fois qu'on voit ce véhicule
                    if track_id not in self.vehicle_trails:
                        # deque avec maxlen=30 : garde seulement les 30 dernières positions
                        self.vehicle_trails[track_id] = deque(maxlen=30)
                    
                    # Récupérer la position précédente du véhicule (s'il y en a une)
                    prev_center = self.vehicle_trails[track_id][-1] if self.vehicle_trails[track_id] else None
                    
                    # Sauvegarder la position actuelle dans l'historique
                    self.vehicle_trails[track_id].append((cx, cy))

                    # LOGIQUE DE COMPTAGE : détection du franchissement de ligne
                    if track_id not in self.counted_ids and prev_center:
                        prev_y = prev_center[1]  # Position Y précédente
                        
                        # Le véhicule a-t-il traversé la ligne ?
                        # Cas 1 : il était au-dessus et maintenant en dessous
                        # Cas 2 : il était en dessous et maintenant au-dessus
                        if (prev_y < line_y and cy >= line_y) or (prev_y > line_y and cy <= line_y):
                            self.total_count += 1
                            self.counts_by_class[vehicle_type] += 1
                            self.counted_ids.add(track_id)  # Marquer comme compté
                            
                            # Feedback visuel : la ligne devient verte brièvement
                            cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 5)

                    # Dessiner le rectangle autour du véhicule (couleur orange)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 191, 0), 2)
                    
                    # Afficher le type de véhicule et son ID au-dessus du rectangle
                    cv2.putText(frame, f"{vehicle_type} [{track_id}]", (int(x1), int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Dessiner la ligne de comptage (rouge par défaut)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        return frame

# INTERFACE PRINCIPALE
st.title("Traffic counter")
st.markdown("""
Cette application utilise **YOLOv8** pour détecter les véhicules, et un logique
pour les suivre et les compter. Chargez une vidéo pour commencer l'analyse.
""")

# Création de 5 colonnes pour afficher les statistiques
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
st_total = kpi1.empty()
st_car = kpi2.empty()
st_moto = kpi3.empty()
st_bus = kpi4.empty()
st_truck = kpi5.empty()
st_frame = st.empty()

if video_file:
    # Création d'un fichier temporaire car OpenCV ne peut pas lire directement depuis Streamlit
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    # Ouverture de la vidéo avec OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    # Création de l'objet compteur
    counter = TrafficCounter()
    
    st.success("Vidéo chargée avec succès ! Analyse en cours...")
    
    stop_button = st.button("Arrêter l'analyse")
    
    # Boucle principale : traitement image par image
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()  # Lire une image
        if not ret:  # Si plus d'images, fin de la vidéo
            st.info("Fin de la vidéo.")
            break      
        
        # Traiter l'image actuelle
        processed_frame = counter.process_frame(frame, line_pos, conf_threshold)
        
        # Mise à jour des statistiques en temps réel
        st_total.metric(label="Total Véhicules", value=counter.total_count)
        st_car.metric(label="Voitures", value=counter.counts_by_class['Voiture'])
        st_moto.metric(label="Motos", value=counter.counts_by_class['Moto'])
        st_bus.metric(label="Bus", value=counter.counts_by_class['Bus'])
        st_truck.metric(label="Camions", value=counter.counts_by_class['Camion'])
        
        # Affichage de l'image traitée (conversion BGR vers RGB pour Streamlit)
        st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    cap.release()  # Libérer la ressource vidéo
else:
    st.info("Veuillez importer une vidéo depuis le menu de gauche pour commencer.")
    
st.markdown("\n")
st.markdown("### Comment l'app ça marche ?")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    **1. Détection (YOLO)**
    L'IA identifie les objets et encadre chacun par une boîte.
    
    **2. Tracking (Suivi)**
    Chaque objet reçoit un ID pour suivre ses déplacements.
    """)
with col_b:
    st.markdown("""
    **3. Ligne Virtuelle**
    Définie par une coordonnée Y, elle sert de déclencheur.
    
    **4. Logique de Comptage**
    Si un objet traverse la ligne, on incrémente le compteur.
    """)