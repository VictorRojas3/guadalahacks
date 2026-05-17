import serial
import numpy as np
import math
import re
import time
import threading
import config
import api_client

csi_history = []
is_moving = False
is_brusco = False
person_state = "ESTATICO"

def process_serial_csi():
    global is_moving, is_brusco, csi_history, person_state
    try:
        ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=1)
        print(f"Conectado al ESP32 en {config.SERIAL_PORT}")
    except Exception as e:
        print(f"Error abriendo puerto serial: {e}")
        return
    last_packet_time = time.time()
    packet_counter = 0
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "CSI_DATA" in line and "[" in line and "]" in line:
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    csi_string = match.group(1)
                    csi_values = [int(x.strip()) for x in csi_string.split(',') if x.strip().lstrip('-').isdigit()]
                    if len(csi_values) > 100:
                        last_packet_time = time.time()
                        packet_counter += 1
                        raw_iq = csi_values[20:] 
                        amplitudes = []
                        for i in range(0, len(raw_iq) - 1, 2):
                            amp = math.sqrt(raw_iq[i]**2 + raw_iq[i+1]**2)
                            amplitudes.append(amp)
                        total_energy = sum(amplitudes)
                        if total_energy > 0:
                            amplitudes = [amp / total_energy for amp in amplitudes]
                        csi_history.append(amplitudes)
                        if len(csi_history) > config.WINDOW_SIZE:
                            csi_history.pop(0)
                        if len(csi_history) == config.WINDOW_SIZE:
                            matrix = np.array(csi_history)
                            varianza_por_subportadora = np.var(matrix, axis=0)
                            indice_movimiento = np.mean(varianza_por_subportadora) * 1000000
                            if packet_counter % 10 == 0:
                                print(f"Subportadoras: {len(amplitudes)} | Indice: {indice_movimiento:.2f} | Estado: {person_state}")
                            if indice_movimiento > config.THRESHOLD_BRUSCO:
                                if person_state == "CAIDA":
                                    person_state = "CAIDA"
                                    is_moving = True
                                    is_brusco = True
                                    threading.Thread(target=api_client.send_critical_fall_alert, daemon=True).start()
                            elif indice_movimiento > config.THRESHOLD_MOVE:
                                is_moving = True
                                is_brusco = False
                                if person_state != "CAIDA":
                                    person_state = "MOVIMIENTO"
                            else:
                                is_moving = False
                                is_brusco = False
                                if person_state != "CAIDA" and person_state != "FUERA_DE_HABITACION":
                                    person_state = "ESTATICO"
            if time.time() - last_packet_time > 1.5:
                is_moving = False
                is_brusco = False
                csi_history.clear()
                if packet_counter > 0:
                    packet_counter = 0   
        except Exception:
            pass

def trigger_routine_room_exit():
    global person_state, is_moving
    is_moving = False
    person_state = "FUERA_DE_HABITACION"
    print("[EVENTO RUTA] Paciente salió de la habitación. Iniciando monitoreo de rutina habitual...")
    api_client.get_dashboard_data()