import numpy as np
import sounddevice as sd
import librosa
import joblib
from flask import Flask, jsonify
import threading
import time

# --- CONFIGURACIÓN DE FLASK ---
app = Flask(__name__)

# Variable global para guardar el último resultado y que la API lo pueda leer
resultado_actual = {
    "estado": "Iniciando...",
    "confianza": 0,
    "color": "gray"
}

# --- CARGAR EL CEREBRO DE CALLI ---
try:
    clf = joblib.load("modelo_calli_final.pkl")
    print("Modelo cargado con éxito. Calli está listo.")
except:
    print("Error: No encontré el archivo 'modelo_calli_final.pkl'")
    exit()

FS = 22050
DURACION = 4
UMBRAL_RUIDO_FUERTE = 0.7

def extraer_features_vivo(y, sr):
    try:
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) > 0:
            y = librosa.util.normalize(y)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        return np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(delta2, axis=1)
        ])
    except:
        return None

def bucle_ia():
    global resultado_actual
    print("🎙️ Motor de IA activado y escuchando...")
    
    while True:
        # 1. Graba el audio (Prueba bajando DURACION a 3 para más velocidad)
        grabacion = sd.rec(int(DURACION * FS), samplerate=FS, channels=1)
        sd.wait()
        y = grabacion.flatten()
        
        # 2. Análisis de Amplitud
        amplitud_maxima = np.max(np.abs(y))
        hay_ruido_fuerte = amplitud_maxima > UMBRAL_RUIDO_FUERTE
        
        # 3. Lógica de Estados (Voz vs Silencio)
        # 0.04 es un buen umbral para ignorar ruidos lejanos o ventiladores
        UMBRAL_ACTIVIDAD = 0.04 

        if amplitud_maxima > UMBRAL_ACTIVIDAD: 
            feat = extraer_features_vivo(y, FS)
            
            if feat is not None:
                prob = clf.predict_proba(feat.reshape(1, -1))[0]
                umbral_tristeza = 0.70
                
                if prob[1] > umbral_tristeza:
                    estado, color = "ALERTA", "red"
                    conf = round(prob[1] * 100, 1)
                else:
                    estado, color = "NORMAL", "green"
                    conf = round(prob[0] * 100, 1)
                
                resultado_actual = {
                    "estado": estado,
                    "confianza": conf,
                    "ruido_fuerte": bool(hay_ruido_fuerte),
                    "color": color,
                    "timestamp": time.strftime("%H:%M:%S")
                }
                
                msg_ruido = "⚠️ ¡ESTRUENDO!" if hay_ruido_fuerte else ""
                print(f"[{resultado_actual['timestamp']}] {estado} ({conf}%) {msg_ruido}")
        
        else:
            # --- MODO REPOSO / SILENCIO ---
            # Si el ruido es menor al umbral, reseteamos a Silencio
            resultado_actual = {
                "estado": "Silencio",
                "confianza": 0,
                "ruido_fuerte": False,
                "color": "gray",
                "timestamp": time.strftime("%H:%M:%S")
            }
            print(f"[{resultado_actual['timestamp']}] ☁️ Silencio...")

        # Quitamos el sleep o lo bajamos al mínimo para evitar lentitud
        time.sleep(0.1)

# --- RUTAS DE LA API ---
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(resultado_actual)

if __name__ == '__main__':
    # 1. Iniciamos el hilo de la IA para que no bloquee a Flask
    thread_ia = threading.Thread(target=bucle_ia, daemon=True)
    thread_ia.start()
    
    # 2. Iniciamos el servidor Flask
    print("\n API de Calli disponible en http://tu_ip_jetson:5000/status")
    app.run(host='0.0.0.0', port=5000, debug=False)