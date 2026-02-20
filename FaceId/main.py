# pip install mediapipe
# pip install cmake
# pip install dlib-bin
# pip install opencv-python
# pip install deepface tf-keras opencv-python
# pip install opencv-contrib-python
# pip install face_recognition

import cv2
from deepface import DeepFace
import os

# Caminho da foto do "Pai" (Base de dados)
DB_PATH = "db/pai.jpg"

# Inicializar Tracker e Detetor de Faces (Haar para velocidade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
tracker = cv2.legacy.TrackerMOSSE_create()

cap = cv2.VideoCapture(0)
tracking_ativo = False
autorizado = False

print("Sistema 'Filho' iniciado. À procura do Pai...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_pequeno = cv2.resize(frame, (640, 480))
    largura_ecran = frame_pequeno.shape[1]

    if not tracking_ativo:
        # 1. DETETAR QUALQUER CARA
        cinza = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2GRAY)
        caras = face_cascade.detectMultiScale(cinza, 1.1, 5)

        for (x, y, w, h) in caras:
            # Cortar a cara detetada para verificar
            face_crop = frame_pequeno[y:y+h, x:x+w]
            
            try:
                # 2. VERIFICAR SE É O PAI (FACE ID)
                # O DeepFace compara a cara no vídeo com a foto na pasta 'db'
                resultado = DeepFace.verify(face_crop, DB_PATH, enforce_detection=False, model_name="VGG-Face")
                
                if resultado["verified"]:
                    # SE FOR O PAI, ATIVA O TRACKING
                    tracker = cv2.legacy.TrackerMOSSE_create()
                    tracker.init(frame_pequeno, (x, y, w, h))
                    tracking_ativo = True
                    autorizado = True
                    print("Pai identificado! A seguir...")
                    break
                else:
                    cv2.putText(frame_pequeno, "Desconhecido - Acesso Negado", (x, y-10), 1, 1, (0, 0, 255), 2)
            except Exception as e:
                continue
    else:
        # 3. TRACKING EXCLUSIVO DO PAI
        sucesso, bbox = tracker.update(frame_pequeno)
        if sucesso:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame_pequeno, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Cálculo de Erro para o motor
            centro_x = x + (w / 2)
            erro_x = centro_x - (largura_ecran / 2)
            
            cv2.putText(frame_pequeno, f"PAI: Seguindo | Erro: {int(erro_x)}", (10, 30), 1, 1.2, (0, 255, 0), 2)
        else:
            tracking_ativo = False
            autorizado = False
            print("Pai saiu de vista. Bloqueando...")

    cv2.imshow("Sistema Pai e Filho", frame_pequeno)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if cv2.waitKey(1) & 0xFF == ord('r'): tracking_ativo = False

cap.release()
cv2.destroyAllWindows()