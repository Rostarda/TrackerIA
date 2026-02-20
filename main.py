import os
import cv2
import time
import threading
from deepface import DeepFace

# Motor TensorFlow em modo silencioso
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DB_FOLDER = "db"
if not os.path.exists(DB_FOLDER): os.makedirs(DB_FOLDER)

identificado = False
processando_ia = False
# Armazena a posição da cara detetada para o tracker usar
posicao_face = None 

def verificar_face_thread(face_img, db_path):
    global identificado, processando_ia
    try:
        arquivos = [f for f in os.listdir(db_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if arquivos:
            foto_ref = os.path.join(db_path, arquivos[0])
            res = DeepFace.verify(face_img, foto_ref, model_name="VGG-Face", 
                                  enforce_detection=False, detector_backend="opencv")
            if res["verified"] or res["distance"] < 0.6:
                identificado = True
    except:
        pass
    processando_ia = False

def criar_tracker():
    """Tenta criar o tracker de várias formas para evitar o erro de AttributeError"""
    try:
        return cv2.TrackerCSRT.create() # Forma moderna
    except AttributeError:
        try:
            return cv2.legacy.TrackerCSRT_create() # Forma legacy
        except AttributeError:
            return cv2.TrackerKCF_create() # Alternativa ultra-rápida se o CSRT falhar

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

tracker = None
tracking_ativo = False
prev_frame_time = 0

print("Sistema Iniciado. Se detetar a face, tentará validar...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    
    frame_display = cv2.resize(frame, (640, 480))

    if not tracking_ativo:
        cinza = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
        caras = face_cascade.detectMultiScale(cinza, 1.1, 5)

        for (x, y, w, h) in caras:
            # Retângulo de busca (Vermelho)
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame_display, "VALIDANDO...", (x, y-10), 1, 1, (0, 0, 255), 2)

            if not processando_ia:
                processando_ia = True
                face_crop = frame_display[y:y+h, x:x+w].copy()
                posicao_face = (x, y, w, h)
                threading.Thread(target=verificar_face_thread, args=(face_crop, DB_FOLDER)).start()

            # SE A THREAD DISSER QUE ÉS TU
            if identificado:
                tracker = criar_tracker()
                tracker.init(frame_display, posicao_face)
                tracking_ativo = True
                identificado = False 
                break
    else:
        # SEGUIMENTO (Verde)
        sucesso, bbox = tracker.update(frame_display)
        if sucesso:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame_display, p1, p2, (0, 255, 0), 3)
            cv2.putText(frame_display, "PAI CONFIRMADO", (p1[0], p1[1]-10), 1, 1, (0, 255, 0), 2)
        else:
            tracking_ativo = False
            tracker = None

    # UI
    cv2.putText(frame_display, f"FPS: {int(fps)}", (520, 30), 1, 1.5, (255, 255, 0), 2)
    cv2.imshow("IA Tracker Pro", frame_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('r'): 
        tracking_ativo = False
        tracker = None

cap.release()
cv2.destroyAllWindows()