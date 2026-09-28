import sys, os, cv2, torch, time #dosya yolları ve sistem işlemleri (sys,os), kamera ve görüntü işleme (cv2)
import torch.nn as nn #yapay zeka modeli, nn-> sinir katmanları
import numpy as np #matematik işlemleri
from collections import deque #geçmiş tahminleri tutmak (stabil sonuç için)
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFrame #Arayüz (GUI) oluşturmak için
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPainterPath
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon #ikon
from torchvision import transforms, models #Hazır model (EfficientNet) ve görüntü dönüşümleri
from PIL import Image
import mediapipe as mp #Yüz tespiti
import pygame  #Ses çalmak için

#PATH
def resource_path(relative_path): #exe yapmak için dosya yolu ayarı
    try:
        base_path = sys._MEIPASS
    except:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

#SES SİSTEMİ
pygame.mixer.init() #ses sistemini başlatır

def duygu_sesi_cal(duygu_adi):
    ses_yolu = resource_path(f"sesler/{duygu_adi}.mp3")
    try:
        if os.path.exists(ses_yolu):
            pygame.mixer.music.load(ses_yolu)
            pygame.mixer.music.play()
    except Exception as e:
        print(f"Ses çalınamadı: {e}")

#MODEL
class_names = ['sinirli', 'iğrenme', 'korku', 'mutlu', 'doğal', 'üzgün', 'şaşkın'] #model sınıfları
MODEL_PATH = resource_path("model_dosyasi.pth") #model dosyası
IMG_SIZE = 300 #görüntü boyutu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #cihaz seçimi
print("Device:",device)

model = models.efficientnet_b3(weights=None) #EfficientNet B-3 Modeli
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device,weights_only=True))
except:
    print("Model dosyası bulunamadı, lütfen yolu kontrol edin.")
model.to(device)
model.eval() #tahmin modu

transform = transforms.Compose([ #modelin isteğine göre görsel boyutları güncellenir
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#MEDIAPIPE
mp_face = mp.solutions.face_detection #kamerada yüz bulunur
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

#RENKLER
emotion_colors = {
    'sinirli': QColor(239, 68, 68),
    'iğrenme': QColor(34, 197, 94),
    'korku': QColor(168, 85, 247),
    'mutlu': QColor(251, 191, 36),
    'doğal': QColor(148, 163, 184),
    'üzgün': QColor(59, 130, 246),
    'şaşkın': QColor(236, 72, 153)
}

emotion_emojis = {
    'sinirli': '😠',
    'iğrenme': '🤢',
    'korku': '😨',
    'mutlu': '😄',
    'doğal': '😐',
    'üzgün': '😢',
    'şaşkın': '😲'
}
#EMOJİ PANEL
class EmojiPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(140)
        self.setFixedHeight(550)
        self.setStyleSheet("background-color:#020617;")
        self.active_index = -1

    def update_active(self, idx): #aktif duyguyu değiştirir
        self.active_index = idx
        self.update()

    def draw_glow(self, painter, x, y, r, color): #seçili duygu etrafına ışık efekti
        for i in range(6, 0, -1):
            alpha = 25 * i
            glow_color = QColor(color.red(), color.green(), color.blue(), alpha)
            painter.setBrush(glow_color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x - r - i * 3, y - r - i * 3, (r + i * 3) * 2, (r + i * 3) * 2)

    def paintEvent(self, event): #tüm emojiler çizilir
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        y_offset = 40
        spacing = 70
        for i, emotion in enumerate(class_names):
            y = y_offset + i * spacing
            color = emotion_colors[emotion]
            if i == self.active_index:
                self.draw_glow(painter, 45, y + 25, 25, color)
                painter.setBrush(color)
            else:
                painter.setBrush(QColor(30, 41, 59))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(21, y, 50, 50)
            painter.setFont(QFont("Segoe UI Emoji", 20))
            painter.setPen(Qt.white)
            painter.drawText(25, y + 35, emotion_emojis[emotion])
            painter.setFont(QFont("Arial", 9))
            painter.setPen(color if i == self.active_index else QColor(148, 163, 184))
            painter.drawText(88, y + 30, emotion)

#KAMERA
class CameraCanvas(QWidget): #orta kamera alanı
    def __init__(self):
        super().__init__()
        self.frame = None
        self.current_emotion = 0
        self.confidence = 0

    def clear(self):
        self.frame = None
        self.update()

    def update_data(self, frame, emotion_idx, conf):
        self.frame = frame
        self.current_emotion = emotion_idx
        self.confidence = conf
        self.update()

    def draw_glow(self, painter, x, y, r, color):
        for i in range(6, 0, -1):
            alpha = 20 * i
            glow_color = QColor(color.red(), color.green(), color.blue(), alpha)
            painter.setBrush(glow_color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x - r - i * 4, y - r - i * 4, (r + i * 4) * 2, (r + i * 4) * 2)

    def paintEvent(self, event):
        if self.frame is None: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        cam_r = int(min(w, h) * 0.30)
        img = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(cam_r * 2, cam_r * 2, Qt.KeepAspectRatioByExpanding)
        path = QPainterPath()
        path.addEllipse(cx - cam_r, cy - cam_r, cam_r * 2, cam_r * 2)
        painter.setClipPath(path)
        painter.drawPixmap(cx - cam_r, cy - cam_r, pix)
        painter.setClipping(False)
        R = int(min(w, h) * 0.41)
        base_r = int(min(w, h) * 0.07)
        for i, emotion in enumerate(class_names):
            angle = (2 * np.pi / len(class_names)) * i
            x = int(cx + R * np.cos(angle))
            y = int(cy + R * np.sin(angle))
            if i == self.current_emotion:
                color = emotion_colors[emotion]
                self.draw_glow(painter, x, y, base_r, color)
                r = int(base_r + self.confidence * 30)
                painter.setBrush(color)
            else:
                r = base_r
                painter.setBrush(QColor(100, 116, 139, 120))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x - r, y - r, r * 2, r * 2)
            painter.setPen(Qt.white)
            painter.setFont(QFont("Arial", 9))
            painter.drawText(x - 20, y + 4, emotion)

#APP
class App(QWidget):  # ana uygulama
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(resource_path("icon.ico"))) #ikon
        self.setWindowTitle("AI Duygu Analiz Sistemi")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color:#0f172a;color:white;")

        # Bileşenler
        self.canvas = CameraCanvas()
        self.emoji_panel = EmojiPanel()

        # --- SOL TARAF DİZİLİMİ ---
        left_side_layout = QVBoxLayout()
        left_side_layout.setContentsMargins(10, 10, 0, 10)  # Üst ve soldan 10px, sağ 0, alt 10px

        # Emoji Paneli Üst Köşeye
        left_side_layout.addWidget(self.emoji_panel, alignment=Qt.AlignTop | Qt.AlignLeft)

        # Esnek boşluk (Panelleri birbirinden uzaklaştırır)
        left_side_layout.addStretch()

        # Kontrol Paneli (SOL ALT KUTU)
        self.control_panel = QFrame()
        self.control_panel.setFixedSize(220, 180)
        self.control_panel.setStyleSheet("""
            QFrame {
                background: #1e293b;
                border-radius: 15px;
                border: 1px solid #334155;
            }
            QLabel { border: none; background: transparent; }
            QPushButton { background: #334155; border-radius: 6px; font-weight: bold; color: white; }
            QPushButton:hover { background: #475569; }
        """)
        cp_layout = QVBoxLayout(self.control_panel)
        self.emotion_label = QLabel("Bekleniyor...")
        self.emotion_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.emotion_label.setAlignment(Qt.AlignCenter)
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setAlignment(Qt.AlignCenter)
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ BAŞLAT")
        self.stop_btn = QPushButton("⏹ DURDUR")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        cp_layout.addWidget(self.emotion_label)
        cp_layout.addWidget(self.fps_label)
        cp_layout.addLayout(btn_layout)

        left_side_layout.addWidget(self.control_panel, alignment=Qt.AlignBottom | Qt.AlignLeft)

        # --- SAĞ TARAF DİZİLİMİ ---
        right_side_layout = QVBoxLayout()
        right_side_layout.setContentsMargins(0, 0, 10, 10)  # Sağ ve alttan 10px pay

        # Kamera Alanı
        right_side_layout.addWidget(self.canvas, stretch=10)

        # Sağ Alt Analiz Kutusu
        self.result_box = QFrame()
        self.result_box.setFixedSize(220, 180)
        self.result_box.setStyleSheet("background:#1e293b; border-radius:20px; border: 2px solid #334155;")
        rb_layout = QVBoxLayout(self.result_box)
        rb_layout.setAlignment(Qt.AlignCenter)

        self.inner_emoji_box = QFrame()
        self.inner_emoji_box.setFixedSize(100, 100)
        self.inner_emoji_box.setStyleSheet("background:#0f172a; border-radius:15px; border:none;")
        self.inner_layout = QVBoxLayout(self.inner_emoji_box)
        self.emoji_label_res = QLabel("🔍")
        self.emoji_label_res.setAlignment(Qt.AlignCenter)
        self.emoji_label_res.setFont(QFont("Segoe UI Emoji", 40))
        self.emoji_label_res.setStyleSheet("border:none; background:transparent;")
        self.inner_layout.addWidget(self.emoji_label_res)

        self.name_label = QLabel("Analiz Bekleniyor...")
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.name_label.setStyleSheet("border:none; margin-top:10px;")

        rb_layout.addWidget(self.inner_emoji_box, alignment=Qt.AlignCenter)
        rb_layout.addWidget(self.name_label)

        right_side_layout.addWidget(self.result_box, alignment=Qt.AlignBottom | Qt.AlignRight)

        # --- ANA LAYOUT BİRLEŞTİRME ---
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addLayout(left_side_layout, 1)
        main_layout.addLayout(right_side_layout, 10)

        # Sinyaller
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

        # Değişkenler
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.prev_time = 0
        self.frame_count = 0
        self.buffer = deque(maxlen=8)
        self.last_detected_emotion = "doğal"

    def start(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.emoji_label_res.setText("🔍")
        self.name_label.setText("Analiz Bekleniyor...")
        self.result_box.setStyleSheet("background:#1e293b; border-radius:20px; border: 2px solid #334155;")

    def stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        emotion = self.last_detected_emotion
        emoji = emotion_emojis.get(emotion, "😐")
        color = emotion_colors[emotion].name()
        self.emoji_label_res.setText(emoji)
        self.name_label.setText(emotion.upper())
        self.name_label.setStyleSheet(f"color:{color}; border:none; font-weight:bold;")
        self.result_box.setStyleSheet(f"background:#1e293b; border-radius:20px; border: 2px solid {color};")
        duygu_sesi_cal(emotion)

    def update_frame(self):
        if self.cap is None: return
        ret, frame = self.cap.read()
        if not ret: return
        self.frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
                face = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if face.size > 0 and self.frame_count % 3 == 0:
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    tensor = transform(face_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
                    self.buffer.append(probs)
        if len(self.buffer) > 0:
            avg = np.mean(self.buffer, axis=0)
            idx = np.argmax(avg)
            conf = float(avg[idx])
            self.last_detected_emotion = class_names[idx]
            self.emotion_label.setText(f"{class_names[idx].upper()} (%{int(conf * 100)})")
            self.canvas.update_data(rgb, idx, conf)
            self.emoji_panel.update_active(idx)
        now = time.time()
        fps = 1 / (now - self.prev_time) if self.prev_time else 0
        self.prev_time = now
        self.fps_label.setText(f"FPS: {int(fps)}")

#RUN
app = QApplication(sys.argv)
window = App()
window.show()
sys.exit(app.exec_())