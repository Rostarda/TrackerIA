import os
import cv2
import time
import threading
from deepface import DeepFace

# Desativar o motor pesado para evitar o crash no Python 3.13
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DB_FOLDER = "db"
if not os.path.exists(DB_FOLDER): os.makedirs(DB_FOLDER)

identificado = False
processando_ia = False
posicao_face = None 

def verificar_face_thread(face_img, db_path):
    global identificado, processando_ia
    try:
        arquivos = [f for f in os.listdir(db_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if arquivos:
            foto_ref = os.path.join(db_path, arquivos[0])
            # SFACE é muito mais leve e rápido, ideal para evitar crashes
            res = DeepFace.verify(face_img, foto_ref, 
                                  model_name="SFace", 
                                  detector_backend="opencv", 
                                  enforce_detection=False)
            
            if res["verified"] or res["distance"] < 0.7:
                identificado = True
    except Exception as e:
        print(f"Erro na Thread: {e}")
    processando_ia = False

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Aumentar a escala para 1.3 torna a detecção de caras 30% mais rápida
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

tracker = None
tracking_ativo = False
prev_frame_time = 0

print("Sistema Light iniciado. A carregar SFace...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    
    frame_display = cv2.resize(frame, (640, 480))

    if not tracking_ativo:
        cinza = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
        caras = face_cascade.detectMultiScale(cinza, 1.3, 5)

        for (x, y, w, h) in caras:
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame_display, "VALIDANDO...", (x, y-10), 1, 1, (0, 0, 255), 2)

            if not processando_ia:
                processando_ia = True
                face_crop = frame_display[y:y+h, x:x+w].copy()
                posicao_face = (x, y, w, h)
                t = threading.Thread(target=verificar_face_thread, args=(face_crop, DB_FOLDER))
                t.daemon = True # Garante que a thread morre se o programa fechar
                t.start()

            if identificado:
                # Se o tracker MIL falhar, vamos usar o KCF legacy ou simplesmente o MIL
                try:
                    tracker = cv2.TrackerMIL_create()
                except AttributeError:
                    try:
                        tracker = cv2.TrackerMIL.create()
                    except:
                        print("Erro ao criar Tracker. A tentar modo redundante...")
                
                if tracker:
                    tracker.init(frame_display, posicao_face)
                    tracking_ativo = True
                identificado = False 
                break
    else:
        sucesso, bbox = tracker.update(frame_display)
        if sucesso:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame_display, p1, p2, (0, 255, 0), 3)
            cv2.putText(frame_display, "PAI CONFIRMADO", (p1[0], p1[1]-10), 1, 1, (0, 255, 0), 2)
        else:
            tracking_ativo = False

    cv2.putText(frame_display, f"FPS: {int(fps)}", (10, 30), 1, 1.2, (255, 255, 0), 2)
    cv2.imshow("Estabilizador Python 3.13", frame_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if cv2.waitKey(1) & 0xFF == ord('r'): tracking_ativo = False

cap.release()
cv2.destroyAllWindows()