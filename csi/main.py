from vpython import *
import threading
import numpy as np
import csi_processor
import api_client

scene = canvas(title='Radar CSI Wi-Fi - Monitoreo avanzado de adultos mayores', width=800, height=600, background=color.black)
scene.camera.pos = vector(0, 2, 5)
head = sphere(pos=vector(0, 1.5, 0), radius=0.4, color=color.white)
body = box(pos=vector(0, 0, 0), size=vector(1, 2, 0.5), color=color.white)
avatar = compound([head, body])
status_text = label(pos=vector(0, 3, 0), text='SISTEMA INICIALIZADO', height=22, color=color.yellow, box=False)


def keyboard_cheat_codes(evt):
    key = evt.key.lower()    
    if key == 'q':
        print("\n[HACK DEMO] Tecla 'Q' detectada. Forzando umbral > 6 (Movimiento brusco / Caída).")
        csi_processor.person_state = "CAIDA"
        csi_processor.is_moving = True
        csi_processor.is_brusco = True
        threading.Thread(target=api_client.send_critical_fall_alert, daemon=True).start()
    elif key == 'e':
        print("\n[HACK DEMO] Tecla 'E' detectada. Forzando evento de salida de habitación.")
        csi_processor.trigger_routine_room_exit()

scene.bind('keydown', keyboard_cheat_codes)
serial_thread = threading.Thread(target=csi_processor.process_serial_csi, daemon=True)
serial_thread.start()
wobble_angle = 0
while True:
    rate(30)
    state = csi_processor.person_state
    if state == "CAIDA":
        status_text.text = "¡MOVIMIENTO BRUSCO / POSIBLE CAÍDA DETECTADA!\n[POST enviado a /api/alerts]"
        status_text.color = color.red
        avatar.color = color.red
        avatar.axis = vector(0, 1, 0)
        avatar.up = vector(-1, 0, 0)
    elif state == "FUERA_DE_HABITACION":
        status_text.text = "PACIENTE SE ENCUENTRA FUERA DE LA HABITACIÓN\n[Estado: Monitoreando Rutina Habitual]"
        status_text.color = color.blue
        avatar.color = color.blue
        avatar.axis = vector(1, 0, 0)
        avatar.up = vector(0, 1, 0)
    elif csi_processor.is_moving:
        status_text.text = "PACIENTE EN MOVIMIENTO (Rutina Normal)"
        status_text.color = color.green
        avatar.color = color.green
        avatar.axis = vector(1, 0, 0)
        avatar.up = vector(0, 1, 0)
        wobble_angle += 0.2
        avatar.rotate(angle=0.05 * np.sin(wobble_angle), axis=vector(0,0,1))
    else:
        status_text.text = "PERSONA ESTÁTICA / HABITACIÓN EN SILENCIO"
        status_text.color = color.orange
        avatar.color = color.orange
        avatar.axis = vector(1, 0, 0)
        avatar.up = vector(0, 1, 0)
        wobble_angle = 0