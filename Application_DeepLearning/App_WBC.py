import sys
import os
import numpy as np

from cv2 import imread, resize, cvtColor, addWeighted
from cv2 import COLOR_BGR2RGB, INTER_NEAREST

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QListWidget, QLabel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ImageLister(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('CNN WBC App')
        self.setGeometry(100, 100, 800, 600)

        # Initialisation des widgets
        self.layout = QVBoxLayout(self)
        self.button_layout = QHBoxLayout()

        self.choose_folder_button = QPushButton('IMG Dir')
        self.choose_model_button = QPushButton('CNN Model')
        self.predict_button = QPushButton('Apply Pred.')
        self.image_list_widget = QListWidget()
        self.display_label = QLabel('Sélectionnez une image pour l\'afficher')
        self.display_label.setAlignment(Qt.AlignCenter)

        # Réduire la hauteur de la liste d'images
        self.image_list_widget.setFixedHeight(100)

        # Ajouter les widgets dans la disposition
        self.button_layout.addWidget(self.choose_folder_button)
        self.button_layout.addWidget(self.choose_model_button)
        self.button_layout.addWidget(self.predict_button)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.image_list_widget)  # Directement la liste sans scroll area
        self.layout.addWidget(self.display_label)

        # Connexion des événements
        self.choose_folder_button.clicked.connect(self.open_folder_dialog)
        self.choose_model_button.clicked.connect(self.open_model_dialog)
        self.predict_button.clicked.connect(self.apply_prediction)
        self.image_list_widget.itemClicked.connect(self.display_image)

        # Répertoire par défaut (vide au départ)
        self.image_folder = None
        self.model = None

    def set_button_background_color(self, button, color):
        """Change la couleur de fond du bouton."""
        button.setStyleSheet(f"background-color: {color};")

    def open_folder_dialog(self):
        """Ouvre une boîte de dialogue pour choisir un répertoire."""
        folder = QFileDialog.getExistingDirectory(self, 'Choisir un répertoire', '')
        if folder:
            self.image_folder = folder
            self.load_images_from_folder(folder)
            # Changer la couleur de fond du bouton pour indiquer que le répertoire est sélectionné
            self.set_button_background_color(self.choose_folder_button, "#1E3A5F")

    def load_images_from_folder(self, folder):
        """Charge les images du répertoire sélectionné."""
        self.image_list_widget.clear()
        # Récupérer tous les fichiers d'image dans le répertoire
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        for filename in os.listdir(folder):
            if any(filename.lower().endswith(ext) for ext in supported_formats):
                self.image_list_widget.addItem(filename)

    def open_model_dialog(self):
        """Ouvre une boîte de dialogue pour choisir un fichier modèle H5."""
        model_file, _ = QFileDialog.getOpenFileName(self, 'Choisir un modèle H5', '', 'Fichiers H5 (*.h5)')
        if model_file:
            self.model = load_model(model_file)
            print(f"Modèle chargé depuis {model_file}")
            
            # Récupérer la taille d'entrée attendue par le modèle
            input_shape = self.model.input_shape
            self.img_size = input_shape[1:3]  # On prend les dimensions h et w, pas le batch size ni les canaux
            
            print(f"Dimensions d'entrée du modèle : {self.img_size}")
            
            # Changer la couleur de fond du bouton pour indiquer que le modèle est chargé
            self.set_button_background_color(self.choose_model_button, "#1E3A5F")

    def display_image(self, item):
        """Affiche l'image sélectionnée."""
        if self.image_folder:
            image_path = os.path.join(self.image_folder, item.text())
            self.display_selected_image(image_path)

    def display_selected_image(self, image_path):
        """Affiche l'image sélectionnée avant la prédiction en la redimensionnant par 5."""
        # Charger l'image
        img = imread(image_path)
        
        # Redimensionner l'image par un facteur de 4
        h, w, _ = img.shape
        new_h = h // 5
        new_w = w // 5
        img_resized = resize(img, (new_w, new_h))

        # Convertir l'image en format RGB
        img_resized = cvtColor(img_resized, COLOR_BGR2RGB)
        
        # Convertir l'image en QImage
        height, width, channel = img_resized.shape
        bytes_per_line = 3 * width
        qimg = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convertir en QPixmap et afficher dans QLabel
        pixmap = QPixmap.fromImage(qimg)
        self.display_label.setPixmap(pixmap)
        self.display_label.setText('')  # Supprimer le texte si une image est affichée

    def apply_prediction(self):
        """Applique la prédiction CNN sur l'image sélectionnée."""
        try:
            if self.image_folder and self.model:
                # Vérifier si une image a été sélectionnée dans la liste
                current_item = self.image_list_widget.currentItem()
                if not current_item:
                    # Si aucune image n'est sélectionnée, afficher un message d'erreur
                    self.display_label.setText("Veuillez sélectionner une image dans la liste.")
                    return

                # Vérifier si un thread est déjà en cours d'exécution
                if hasattr(self, 'predict_thread') and self.predict_thread.isRunning():
                    return  # Ne pas démarrer un autre thread si un thread est déjà en cours

                # Cacher le bouton de prédiction (désactivé) sans barre de progression
                self.predict_button.setEnabled(False)

                # Lancer la prédiction dans un thread séparé pour éviter de bloquer l'interface
                image_path = os.path.join(self.image_folder, current_item.text())
                self.predict_thread = PredictThread(image_path, self.model, self.img_size)
                self.predict_thread.prediction_done.connect(self.display_prediction_result)
                self.predict_thread.finished.connect(self.enable_predict_button)  # Réactiver le bouton une fois le thread terminé
                self.predict_thread.start()
            else:
                self.display_label.setText("Veuillez charger un modèle et un répertoire d'images.")
        except Exception as e:
            print(f'erreur: {e}')

    def display_prediction_result(self, result_image):
        """Affiche le résultat de la prédiction dans la fenêtre."""
        # Redimensionner l'image prédite aux mêmes dimensions que l'image avant la prédiction
        resized_result_image = resize(result_image, (self.display_label.pixmap().width(), self.display_label.pixmap().height()))

        # Afficher l'image redimensionnée
        self.display_predicted_image(resized_result_image)

        # Réactiver le bouton de prédiction
        self.predict_button.setEnabled(True)

    def display_predicted_image(self, predicted_img):
        """Affiche l'image prédite dans l'interface."""
        # Convertir l'image prédite en QImage
        height, width, channel = predicted_img.shape
        bytes_per_line = 3 * width
        # Assurez-vous que l'image est sous forme de bytearray et non pas en memoryview
        predicted_img = np.array(predicted_img, dtype=np.uint8)
        qimg = QImage(predicted_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convertir en QPixmap et afficher dans QLabel
        pixmap = QPixmap.fromImage(qimg)
        self.display_label.setPixmap(pixmap)
        self.display_label.setText('')  # Supprimer le texte si une image est affichée

    def enable_predict_button(self):
        """Réactive le bouton de prédiction une fois que la prédiction est terminée."""
        self.predict_button.setEnabled(True)


class PredictThread(QThread):
    prediction_done = pyqtSignal(np.ndarray)

    def __init__(self, image_path, model, img_size):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.img_size = img_size

    def run(self):
        """Exécute la prédiction CNN."""
        self.visualize_surface_prediction(self.image_path)

    def visualize_surface_prediction(self, image_path, percentile=98.5):
        """Effectue la prédiction et génère le résultat à afficher."""
        try:
            # Préparer l'image pour le modèle
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img) / 255.0
            pred = self.model.predict(np.expand_dims(img_array, axis=0))[0]

            # Calcul du seuil basé sur le percentile
            threshold = np.percentile(pred, percentile)
            pred_bin = (pred > threshold).astype(np.uint8)

            # Charger l'image originale et redimensionner la prédiction
            original_img = cvtColor(imread(image_path), COLOR_BGR2RGB)
            pred_bin_resized = resize(pred_bin, (original_img.shape[1], original_img.shape[0]), interpolation=INTER_NEAREST)

            # Créer un masque de couleur verte (RGBA) avec opacité 50% pour les zones détectées
            overlay = np.zeros_like(original_img)
            overlay[pred_bin_resized == 1] = [0, 255, 0]  # Vert pur pour les zones détectées
            blended_img = addWeighted(original_img, 1.0, overlay, 0.55, 0)

            # Retourner l'image prédite
            self.prediction_done.emit(blended_img)
        except Exception as e:
            print(f'{e}')


# Rediriger stdout vers un fichier
sys.stdout = open('output.log', 'w')

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        # Appliquer le style sombre
        app.setStyleSheet("""
            QWidget {
                background-color: #2e2e2e;
                color: white;
            }
            QPushButton {
                background-color: #444444;
                color: white;
                border: 1px solid white;
                padding: 5px 10px;
                font-size: 13px;  /* Taille de la police des boutons */
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QListWidget {
                background-color: #333333;
                color: white;
                border: 1px solid white;
            }
            QLabel {
                color: white;
            }
        """)

        window = ImageLister()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        # Log ou afficher l'erreur
        print(f"Une erreur est survenue : {str(e)}", file=sys.stdout)