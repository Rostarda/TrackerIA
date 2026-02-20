import os
import cv2
import time
import threading
from deepface import DeepFace
from plyer import notification

# Configurações de sistema
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DB_FOLDER = "db"
identificado = False
processando_ia = False

def enviar_aviso_windows():
    try:
        notification.notify(
            title='Alerta de Segurança',
            message='O Pai desapareceu do mapa!',
            timeout=3
        )
    except: pass

def verificar_face_thread(face_img, db_path):
    global identificado, processando_ia
    try:
        arquivos = [f for f in os.listdir(db_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if arquivos:
            foto_ref = os.path.join(db_path, arquivos[0])
            res = DeepFace.verify(face_img, foto_ref, model_name="SFace", 
                                  detector_backend="opencv", enforce_detection=False)
            identificado = res["verified"]
    except: identificado = False
    processando_ia = False

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

tracker = None
tracking_ativo = False
last_x, last_y = 0, 0
tempo_parado = 0

print("Sistema Anti-Erro iniciado...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_display = cv2.resize(frame, (640, 480))
    
    if not tracking_ativo:
        cinza = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
        caras = face_cascade.detectMultiScale(cinza, 1.3, 5)

        for (x, y, w, h) in caras:
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            if not processando_ia:
                processando_ia = True
                face_crop = frame_display[y:y+h, x:x+w].copy()
                threading.Thread(target=verificar_face_thread, args=(face_crop, DB_FOLDER), daemon=True).start()

            if identificado:
                tracker = cv2.TrackerMIL.create()
                tracker.init(frame_display, (x, y, w, h))
                tracking_ativo = True
                identificado = False
                tempo_parado = time.time()
                break
    else:
        sucesso, bbox = tracker.update(frame_display)
        
        if sucesso:
            x, y, w, h = [int(v) for v in bbox]
            
            # --- FILTRO DE "OBJETO PARADO" ---
            # Se o retângulo não se mexer mais de 5 pixels, começa a contar tempo
            if abs(x - last_x) < 5 and abs(y - last_y) < 5:
                if time.time() - tempo_parado > 5.0: # 5 segundos parado = erro
                    sucesso = False 
            else:
                tempo_parado = time.time()
            
            last_x, last_y = x, y

            # --- FILTRO DE SAÍDA DE ECRÃ ---
            if x < 10 or y < 10 or x+w > 630 or y+h > 470:
                sucesso = False

            if sucesso:
                cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(frame_display, "PAI SEGUIDO", (x, y-10), 1, 1, (0, 255, 0), 2)
            else:
                tracking_ativo = False
                threading.Thread(target=enviar_aviso_windows).start()
        else:
            tracking_ativo = False
            threading.Thread(target=enviar_aviso_windows).start()

    cv2.imshow("Tracker Inteligente", frame_display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()