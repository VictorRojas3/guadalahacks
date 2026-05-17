import requests
import config

def send_critical_fall_alert():
    url = f"{config.API_BASE_URL}/ingest/presence"   
    payload = {
      "sourceId": "Dispositivo-1",
      "personId": "Adulto_001",
      "roomId": "Habitacion_principal",
      "presence": True,
      "zone": "floor",
      "movement": "low",
      "posture": "lying",
      "possibleFall": True,
      "immobileSeconds": 5,
      "confidence": 0.9
    }

    try:
        print(f"[POST] Enviando alerta crítica a {url}...")
        response = requests.post(url, json=payload, timeout=2)
        print(f"[API RESPUESTA] Código de estado: {response.status_code}")
        return True
    except Exception as e:
        print(f"[API ERROR] No se pudo enviar el POST (Modo Demo Offline): {e}")
        return False


def get_current_alerts():
    print("[GET] /api/alerts/current -> Consultando estado de alertas activas.")

def send_caregiver_action():
    print("[POST] /api/alerts/alert_001/actions -> Registrando llamada a cuidador.")

def patch_resolve_alert():
    print("[PATCH] /api/alerts/alert_001 -> Cambiando estado de alerta a 'resolved'.")

def get_dashboard_data():
    print("[GET] /api/dashboard -> Actualizando métricas generales del panel principal.")