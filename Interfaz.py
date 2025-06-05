import sys
import numpy as np
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar, QComboBox
from PySide6.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor
from PySide6.QtCore import Qt, QTimer, QPoint
import tensorflow as tf
from tensorflow.keras.models import load_model

class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clasificador Avanzado de Frutas")
        self.setGeometry(100, 100, 900, 800)
        
        # Configuración inicial
        self.setup_ui()
        self.setup_camera()
        self.load_model()
        
    def setup_ui(self):
        # Estilos CSS mejorados
        self.setStyleSheet("""
            QWidget { 
                background-color: #f5f5f5; 
                color: #333333;
                font-family: Arial;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 15px;
                font-size: 14px;
                border-radius: 4px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLabel {
                font-size: 16px;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                background-color: white;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
            }
        """)

        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Título con estilo mejorado
        title = QLabel("Clasificador de Frutas")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        main_layout.addWidget(title)

        # Layout de dos columnas
        columns_layout = QHBoxLayout()
        
        # Columna izquierda: controles y resultados
        left_column = QVBoxLayout()
        left_column.setSpacing(15)
        
        # Panel de controles
        controls_box = QWidget()
        controls_box.setStyleSheet("background-color: white; border-radius: 8px; padding: 15px;")
        controls_layout = QVBoxLayout(controls_box)
        
        # Controles de cámara
        camera_title = QLabel("Controles de Cámara")
        camera_title.setFont(QFont("Arial", 14, QFont.Bold))
        camera_title.setStyleSheet("margin-bottom: 10px;")
        controls_layout.addWidget(camera_title)
        
        camera_controls = QHBoxLayout()
        
        self.camera_btn = QPushButton("Activar Cámara")
        self.camera_btn.setStyleSheet("Background-Color: black")
        self.camera_btn.clicked.connect(self.toggle_camera)
        camera_controls.addWidget(self.camera_btn)
        
        self.detection_btn = QPushButton("Iniciar Detección")
        self.detection_btn.setStyleSheet("Background-Color: black")
        self.detection_btn.setEnabled(False)
        self.detection_btn.clicked.connect(self.toggle_detection)
        camera_controls.addWidget(self.detection_btn)
        
        controls_layout.addLayout(camera_controls)
        
        # Controles de archivo
        file_title = QLabel("Controles de Archivo")
        file_title.setFont(QFont("Arial", 14, QFont.Bold))
        file_title.setStyleSheet("margin-top: 20px; margin-bottom: 10px;")
        controls_layout.addWidget(file_title)
        
        self.file_btn = QPushButton("Cargar Imagen")
        self.file_btn.setStyleSheet("Background-Color: black")
        self.file_btn.clicked.connect(self.load_image)
        controls_layout.addWidget(self.file_btn)
        
        self.predict_btn = QPushButton("Predecir Imagen")
        self.predict_btn.setStyleSheet("Background-Color: black")
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self.predict_file_image)
        controls_layout.addWidget(self.predict_btn)
        
        left_column.addWidget(controls_box)
        
        # Panel de resultados
        results_box = QWidget()
        results_box.setStyleSheet("background-color: white; border-radius: 8px; padding: 15px;")
        results_layout = QVBoxLayout(results_box)
        
        results_title = QLabel("Resultados de Predicción")
        results_title.setFont(QFont("Arial", 14, QFont.Bold))
        results_title.setStyleSheet("margin-bottom: 10px;")
        results_layout.addWidget(results_title)
        
        self.result_label = QLabel("Esperando análisis...")
        self.result_label.setFont(QFont("Arial", 12))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("min-height: 150px;")
        results_layout.addWidget(self.result_label)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        results_layout.addWidget(self.progress_bar)
        
        left_column.addWidget(results_box)
        
        # Columna derecha: visualización
        right_column = QVBoxLayout()
        
        # Etiqueta para mostrar la imagen/cámara
        self.image_label = CameraLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: #e9ecef;
            border: 2px solid #dee2e6;
            border-radius: 8px;
        """)
        self.image_label.setMinimumSize(640, 480)
        right_column.addWidget(self.image_label)
        
        # Estado de detección
        self.status_label = QLabel("Cámara desactivada")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-style: italic; color: #6c757d; margin-top: 10px;")
        right_column.addWidget(self.status_label)
        
        columns_layout.addLayout(left_column, 1)
        columns_layout.addLayout(right_column, 2)
        
        main_layout.addLayout(columns_layout)
        self.setLayout(main_layout)

    def setup_camera(self):
        # Intentar abrir la cámara
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo abrir la cámara")
            return

        # Configurar temporizador para la cámara
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        
        # Estado inicial
        self.is_camera_active = False
        self.is_detection_active = False
        self.captured_image = None
        self.detection_rect = None
        self.last_prediction = ""

    def load_model(self):
        try:
            # Cargar modelo de clasificación
            self.classifier_model = load_model("modelo.keras")
            self.classes = ['manzana', 'platano']
            
            # Cargar modelo de detección (usaremos un detector simple basado en color)
            self.status_label.setText("Modelos cargados correctamente")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar el modelo: {str(e)}")
            self.classifier_model = None
            self.status_label.setText("Error al cargar modelos")

    def toggle_camera(self):
        if not self.is_camera_active:
            # Activar cámara
            self.camera_timer.start(20)
            self.camera_btn.setText("Desactivar Cámara")
            self.camera_btn.setStyleSheet("Background-Color: black")
            self.detection_btn.setEnabled(True)
            self.file_btn.setEnabled(False)
            self.status_label.setText("Cámara activada - Mostrando vista previa")
            self.is_camera_active = True
        else:
            # Desactivar cámara
            self.camera_timer.stop()
            self.camera_btn.setText("Activar Cámara")
            self.camera_btn.setStyleSheet("Background-Color: black")
            self.detection_btn.setText("Iniciar Detección")
            self.detection_btn.setStyleSheet("Background-Color: black")
            self.detection_btn.setEnabled(False)
            self.file_btn.setEnabled(True)
            self.is_camera_active = False
            self.is_detection_active = False
            self.status_label.setText("Cámara desactivada")
            
            # Mostrar último frame capturado
            if self.captured_image is not None:
                self.display_image(self.captured_image)

    def toggle_detection(self):
        if not self.is_detection_active:
            # Iniciar detección
            self.detection_btn.setText("Detener Detección")
            self.detection_btn.setStyleSheet("Background-Color: black")
            self.status_label.setText("Modo detección activo - Buscando frutas...")
            self.is_detection_active = True
        else:
            # Detener detección
            self.detection_btn.setText("Iniciar Detección")
            self.detection_btn.setStyleSheet("Background-Color: black")
            self.status_label.setText("Cámara activada - Mostrando vista previa")
            self.is_detection_active = False

    def update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return
            
        # Guardar frame actual
        self.current_frame = frame.copy()
        
        if self.is_detection_active:
            # Realizar detección de frutas
            processed_frame, fruit_roi = self.detect_fruit(frame)
            
            if fruit_roi is not None:
                # Clasificar la fruta detectada
                self.classify_fruit(fruit_roi)
                frame = processed_frame
            else:
                self.result_label.setText("Buscando frutas...")
        else:
            # Solo mostrar vista previa
            processed_frame = frame
        
        # Mostrar el frame en la interfaz
        self.display_image(processed_frame)

    def detect_fruit(self, frame):
        """Detecta frutas en el frame usando técnicas de visión por computadora"""
        # Convertir a espacio de color HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Definir rangos de color para frutas comunes
        # (Estos valores deberían ajustarse para tu entorno específico)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        
        # Crear máscaras para diferentes colores de frutas
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combinar máscaras
        mask = cv2.bitwise_or(mask_red, mask_yellow)
        mask = cv2.bitwise_or(mask, mask_green)
        
        # Operaciones morfológicas para limpiar la máscara
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fruit_roi = None
        if contours:
            # Encontrar el contorno más grande (suponemos que es la fruta)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Filtrar por tamaño mínimo
            if area > 1000:  # Área mínima para considerar como fruta
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Guardar región de interés para clasificación
                fruit_roi = frame[y:y+h, x:x+w]
                
                # Dibujar rectángulo alrededor de la fruta
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Guardar coordenadas para dibujar en la interfaz
                self.detection_rect = (x, y, w, h)
                
                # Actualizar estado
                self.status_label.setText(f"Fruta detectada ({area} px)")
            else:
                self.detection_rect = None
                self.status_label.setText("Buscando frutas...")
        else:
            self.detection_rect = None
            self.status_label.setText("Buscando frutas...")
        
        return frame, fruit_roi

    def classify_fruit(self, roi):
        """Clasifica la región de interés que contiene una fruta"""
        if roi is None or roi.size == 0:
            return
            
        try:
            # Preprocesar la imagen para el modelo
            roi_resized = cv2.resize(roi, (128, 128))
            roi_normalized = roi_resized.astype(np.float32) / 255.0
            roi_input = np.expand_dims(roi_normalized, axis=0)
            
            # Realizar predicción
            prediction = self.classifier_model.predict(roi_input)[0]
            
            # Formatear resultados
            result_text = "Resultado:\n"
            max_prob = 0
            max_class = ""
            
            for class_name, prob in zip(self.classes, prediction):
                percentage = round(prob * 100, 2)
                result_text += f"{class_name}: {percentage}%\n"
                
                if prob > max_prob:
                    max_prob = prob
                    max_class = class_name
            
            # Solo actualizar si tenemos una predicción confiable
            if max_prob > 0.7:
                self.last_prediction = f"Fruta detectada: {max_class} ({max_prob*100:.1f}%)"
                self.result_label.setText(result_text)
                self.status_label.setText(self.last_prediction)
        except Exception as e:
            self.result_label.setText(f"Error en clasificación: {str(e)}")

    def display_image(self, image):
        """Muestra una imagen OpenCV en el QLabel"""
        # Convertir de BGR a RGB
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            qt_image = image  # Ya es QImag
            
        # Mostrar en la etiqueta
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.width(), self.image_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def load_image(self):
        if self.is_camera_active:
            self.toggle_camera()  # Desactivar cámara si está activa

        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        if file_name:
            self.captured_image = " "
            # Cargar imagen con OpenCV
            self.captured_image = cv2.imread(file_name)
            
            # Mostrar imagen estática
            self.display_image(self.captured_image)
            
            # Habilitar predicción
            self.predict_btn.setEnabled(True)
            self.predict_btn.setStyleSheet("Background-Color: black")
            self.status_label.setText("Imagen cargada - Lista para análisis")

    def predict_file_image(self):
        if self.captured_image is None:
            QMessageBox.warning(self, "Error", "No hay imagen para analizar")
            return

        # Mostrar barra de progreso
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        try:
            # Asegurarnos de que tenemos un array numpy
            if not isinstance(self.captured_image, np.ndarray):
                # Convertir QImage a array numpy si es necesario
                qimage = self.captured_image
                ptr = qimage.bits()
                ptr.setsize(qimage.byteCount())
                arr = np.array(ptr).reshape(qimage.height(), qimage.width(), 4)  # RGBA
                self.captured_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            
            # Preprocesar la imagen
            image_rgb = cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB)
            processed_image = cv2.resize(image_rgb, (128, 128))
            processed_image = processed_image.astype(np.float32) / 255.0
            processed_image = np.expand_dims(processed_image, axis=0)
            
            self.progress_bar.setValue(30)
            
            # Predecir
            prediction = self.classifier_model.predict(processed_image)[0]
            self.progress_bar.setValue(70)
            
            result_text = "Resultado:\n"
            for class_name, prob in zip(self.classes, prediction):
                percentage = round(prob * 100, 2)
                result_text += f"{class_name}: {percentage}%\n"
            
            self.result_label.setText(result_text)
            self.progress_bar.setValue(100)
            
            # Actualizar estado
            max_index = np.argmax(prediction)
            max_class = self.classes[max_index]
            max_prob = prediction[max_index]
            self.status_label.setText(f"Predicción: {max_class} ({max_prob*100:.1f}%)")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en la predicción: {str(e)}")
        
        self.progress_bar.setVisible(False)

    def closeEvent(self, event):
        if hasattr(self, 'camera') and self.camera.isOpened():
            self.camera.release()
        event.accept()


class CameraLabel(QLabel):
    """QLabel personalizado para mostrar la detección en tiempo real"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.detection_rect = None
    
    def set_detection_rect(self, rect):
        self.detection_rect = rect
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.detection_rect:
            # Obtener las coordenadas del rectángulo
            x, y, w, h = self.detection_rect
            
            # Escalar las coordenadas según el tamaño de la etiqueta
            pixmap = self.pixmap()
            if pixmap:
                label_size = self.size()
                pixmap_size = pixmap.size()
                
                # Calcular relación de aspecto
                scale_w = label_size.width() / pixmap_size.width()
                scale_h = label_size.height() / pixmap_size.height()
                scale = min(scale_w, scale_h)
                
                # Calcular offset para centrar
                offset_x = (label_size.width() - pixmap_size.width() * scale) / 2
                offset_y = (label_size.height() - pixmap_size.height() * scale) / 2
                
                # Ajustar coordenadas del rectángulo
                rect_x = offset_x + x * scale
                rect_y = offset_y + y * scale
                rect_w = w * scale
                rect_h = h * scale
                
                # Dibujar rectángulo
                painter = QPainter(self)
                painter.setPen(QPen(QColor(0, 255, 0), 3))
                painter.drawRect(rect_x, rect_y, rect_w, rect_h)
                
                # Dibujar etiqueta
                painter.setFont(QFont("Arial", 12))
                painter.drawText(rect_x, rect_y - 10, "Fruta detectada")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec())