# 🚦 Système Intelligent de Comptage de Trafic

<p align="center">
  <img src="demo-traffic.gif" width="600" alt="Démo Traffic Counter">
</p>

## • Présentation

Ce projet est une solution de monitoring de trafic routier basée sur l'Intelligence Artificielle. Il permet de détecter, suivre et compter différentes catégories de véhicules (voitures, bus, camions, motos) en temps réel à l'aide d'une ligne virtuelle de franchissement.

L'objectif est d'extraire des données statistiques exploitables à partir de flux vidéos complexes, en combinant la puissance de **YOLOv8** pour la détection et une interface interactive développée avec **Streamlit**.

## • Stack Technique

* **IA & Computer Vision :**  `YOLOv8` (Ultralytics) pour la détection et le tracking.
* **Traitement d'Image :** `OpenCV` pour la manipulation des matrices vidéo et le rendu visuel.
* **Interface Utilisateur :** `Streamlit` pour le dashboard interactif.
* **Logique de Tracking :** `Centroid Tracking` avec gestion d'historique via `collections.deque`.

## • Fonctionnalités Clés

* **Classification Intelligente :** Identification précise basée sur le dataset COCO (Voitures, Motos, Bus, Camions).
* **Suivi Persistant :** Chaque véhicule possède un ID unique pour éviter les doublons lors du comptage.
* **Dashboard Dynamique :** * Réglage du seuil de confiance (Confidence Score) en direct.
* Positionnement ajustable de la ligne de comptage.
* Statistiques (KPIs) mises à jour en temps réel.


* **Feedback Visuel :** Changement de couleur de la ligne lors du franchissement et tracé des trajectoires.

## • Installation & Lancement

1. **Cloner le repository :**
```bash
git clone https://github.com/abdelhakimlaachachi/traffic-counter.git
cd traffic-counter

```


2. **Installer les dépendances :**
```bash
pip install streamlit ultralytics opencv-python numpy

```


3. **Lancer l'application :**
```bash
streamlit run app.py

```



## • Cas d'Utilisation

* Analyse de la densité du trafic urbain.
* Optimisation de la signalisation routière.
* Collecte de données pour les infrastructures de "Smart City".