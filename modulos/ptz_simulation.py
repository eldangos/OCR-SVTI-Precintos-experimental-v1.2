"""GUI-thread simulator. Has no camera, network, OCR or storage dependency."""
from PySide6.QtCore import QTimer
import math


class PtzSimulation:
    def __init__(self, controller, log):
        self.controller = controller
        self.log = log
        self.serial = 0
        self.demo_box = None
        self.shape = [600, 1000, 3]
        # Virtual scene response only. The physical pulse limits stay unchanged.
        self.zoom_gain = 1.4

    def stop(self):
        self.serial += 1
        self.log('SIM · STOP')

    def pulse(self, *, duration_s, **command):
        self.serial += 1
        token = self.serial
        self.log(f'SIM · PTZ {command}, {duration_s:.2f} s')
        def complete():
            if token == self.serial:
                if self.demo_box is not None:
                    x1, y1, x2, y2 = self.demo_box
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    height, width = self.shape[:2]
                    if command.get('pan'):
                        cx -= self.controller.pan_sign * (1 if command['pan'] > 0 else -1) * min(.04*width, .35*(x2-x1), abs(width/2-cx))
                    if command.get('tilt'):
                        cy -= self.controller.tilt_sign * (1 if command['tilt'] > 0 else -1) * min(.05*height, .35*(y2-y1), abs(height/2-cy))
                    scale = 1.
                    if command.get('zoom'):
                        scale = min(1.4, self.zoom_gain ** (duration_s / .12 * abs(command['zoom']) / 15))
                        if command['zoom'] < 0:
                            scale = 1 / scale
                        size = max((x2-x1)/width, (y2-y1)/height)
                        needed = max(0, math.ceil(math.log(self.controller.target_size/size) / math.log(scale))) if scale > 1 else None
                        remaining = min(self.controller.max_zoom_steps - self.controller._zooms + 1,
                                        int((self.controller.max_zoom_s - self.controller._zoom_time + duration_s + 1e-9) / duration_s))
                        self.log(f'SIM · ganancia virtual={scale:.3f}, tamaño={size:.4f}, '
                                 f'objetivo={self.controller.target_size:.4f}, pulsos necesarios={needed}, '
                                 f'disponibles incluido actual={remaining}; presupuesto PTZ sin cambios')
                    hx, hy = (x2-x1)*scale/2, (y2-y1)*scale/2
                    self.demo_box = [cx-hx, cy-hy, cx+hx, cy+hy]
                    self.log(f'SIM · tamaño resultante={max(2*hx/width, 2*hy/height):.4f}, '
                             f'zoom acumulado={self.controller._zoom_time:.3f}/{self.controller.max_zoom_s:.3f} s, '
                             f'pasos={self.controller._zooms}/{self.controller.max_zoom_steps}')
                self.log(f'SIM · STOP confirmado · token {token} · caja {self.demo_box}')
                self.controller.motion_result(dict(token=token, phase='stopped', ok=True))
        QTimer.singleShot(round(duration_s * 1000), complete)
        return token

    def focus(self):
        self.log('SIM · enfoque')
        self.controller.focus_result('Modo AUTO verificado (simulación)')

    def capture(self):
        self.serial += 1
        token = self.serial
        self.log('SIM · captura 101')
        def complete():
            if token == self.serial and self.controller.photo_checked(token, True):
                self.log('SIM · OCR SIMULADO')
                def finish():
                    if token == self.serial:
                        self.log('SIM · guardado virtual; sin lectura auténtica')
                        self.controller.ocr_result(token, success=True)
                        self.log('DEMO VIRTUAL · ' + self.controller.status)
                QTimer.singleShot(300, finish)
        QTimer.singleShot(50, complete)
        return token
