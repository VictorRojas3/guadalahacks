import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- CONFIGURACIÓN DE RUTA ---
# Apuntamos a la carpeta principal
BASE_PATH = "mi_dataset" 

def extraer_features(ruta):
    try:
        y, sr = librosa.load(ruta, duration=4, sr=22050)
        
        # 1. Quitar silencios (Para enfocarnos en la voz)
        y, _ = librosa.effects.trim(y, top_db=20) 
        
        # 2. Normalizar volumen (Para que la distancia al micro no afecte)
        if len(y) > 0:
            y = librosa.util.normalize(y)
        else:
            return None

        # 3. Extraer MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 4. Deltas (Velocidad y Aceleración del habla)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Combinamos todo: Promedios y Variaciones
        return np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(delta2, axis=1)
        ])
    except:
        return None

def cargar_datos(diccionario_carpetas):
    X, y = [], []
    for carpeta, etiqueta in diccionario_carpetas.items():
        # Construimos la ruta: mi_dataset/nombre_carpeta
        ruta_completa = os.path.join(BASE_PATH, carpeta)
        
        if not os.path.exists(ruta_completa):
            print(f"Error: No encuentro {ruta_completa}")
            continue
            
        print(f"Procesando {carpeta}...")
        for archivo in os.listdir(ruta_completa):
            if archivo.endswith(".wav"):
                feat = extraer_features(os.path.join(ruta_completa, archivo))
                if feat is not None:
                    X.append(feat)
                    y.append(etiqueta)
    return np.array(X), np.array(y)

# --- 1. CARGAR DATOS ---
# 0 = Normal/Saludable, 1 = Triste/Alerta
entrenamiento = {"normal_train": 0, "triste_train": 1}
prueba = {"normal_test": 0, "triste_test": 1}

print("Iniciando entrenamiento de Calli con 'mi_dataset'...")
X_train, y_train = cargar_datos(entrenamiento)
X_test, y_test = cargar_datos(prueba)

# --- 2. MODELO ---
if len(X_train) > 0:
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
    ])

    clf.fit(X_train, y_train)

    # --- 3. VALIDACIÓN ---
    y_pred = clf.predict(X_test)
    print(f"\nEPORTE DE PRECISIÓN (Basado en el 4to integrante):")
    print(f"Precisión Global: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("\n--- Clasificación ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Triste"]))

    # --- 4. GUARDAR ---
    joblib.dump(clf, "modelo_calli_final.pkl")
    print("\nModelo 'modelo_calli_final.pkl' generado con éxito.")
else:
    print("Error: No se cargaron datos. Revisa que las carpetas estén en 'mi_dataset'.")