"""Camera-independent, bounded YOLO -> PTZ -> existing capture/OCR coordinator."""
import math
import time

from modulos.yolo_vigilance import covers_frame, overlap


class AutoPtzController:
    RETAIN_S = .35
    FRESH_VISUAL_S = .20
    ACTIVE = {'DETECTADO', 'SIGUIENDO', 'CENTRANDO', 'ACERCANDO', 'ESTABILIZANDO', 'LISTO PARA CAPTURAR', 'CAPTURANDO', 'OCR'}

    def __init__(self, gate, *, pulse, stop, focus, focus_pending, capture, available,
                 capture_allowed=lambda: True, clock=time.monotonic, deadband=.12, target_size=.30, speed=12,
                 zoom_speed=15, pulse_s=.12, max_zoom_s=1.2, max_moves=16,
                 max_zoom_steps=8, max_session_s=35, target_age_s=1.2,
                 settle_s=.8, sharpness=45, photo_sharpness=35, retries=1,
                 pan_sign=1, tilt_sign=-1, physical=False, directions_verified=False,
                 observe_only=False, recheck_s=1.2, flow_max_s=5, retain_s=.55,
                 keep_confidence=.35, appearance_min=.70):
        if not (.20 <= retain_s <= 1 and .20 <= recheck_s < flow_max_s <= 15
                and 0 < keep_confidence <= 1 and 0 < appearance_min <= 1):
            raise ValueError('Parámetros de seguimiento híbrido fuera de rango')
        vals = (deadband, target_size, speed, zoom_speed, pulse_s, max_zoom_s,
                max_moves, max_zoom_steps, max_session_s, target_age_s, settle_s,
                sharpness, photo_sharpness, retries, pan_sign, tilt_sign)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError('Parámetros PTZ no finitos')
        if not (0 < deadband < .5 and 0 < target_size < .8 and 1 <= speed <= 30
                and 1 <= zoom_speed <= 30 and .05 <= pulse_s <= .4
                and 0 < max_zoom_s <= 5 and 1 <= max_moves <= 50
                and 1 <= max_zoom_steps <= 20 and 1 <= max_session_s <= 120
                and .2 <= target_age_s <= 2 and settle_s >= .5
                and sharpness > 0 and photo_sharpness > 0 and 0 <= retries <= 2
                and pan_sign in (-1, 1) and tilt_sign in (-1, 1)):
            raise ValueError('Parámetros PTZ fuera de los límites de seguridad')
        self.gate = gate
        self.pulse, self.stop, self.focus = pulse, stop, focus
        self.focus_pending, self.capture, self.available = focus_pending, capture, available
        self.clock = clock
        self.capture_allowed = capture_allowed
        self.deadband, self.target_size = deadband, target_size
        self.speed, self.zoom_speed, self.pulse_s = int(speed), int(zoom_speed), pulse_s
        self.max_zoom_s, self.max_moves, self.max_zoom_steps = max_zoom_s, int(max_moves), int(max_zoom_steps)
        self.max_session_s, self.target_age_s, self.settle_s = max_session_s, target_age_s, settle_s
        self.sharpness, self.photo_sharpness, self.retries = sharpness, photo_sharpness, int(retries)
        self.pan_sign, self.tilt_sign = pan_sign, tilt_sign
        self.physical = physical
        self.directions_verified = directions_verified
        self.observe_only = observe_only
        self.recheck_s, self.flow_max_s, self.retain_s = recheck_s, flow_max_s, retain_s
        self.keep_confidence, self.appearance_min = keep_confidence, appearance_min
        self.decision = None
        self.enabled = False
        self.state, self.reason = 'VIGILANCIA', 'PTZ automático desactivado'
        self.capture_id = None
        self._token = None
        self._terminal_at = 0
        self._sequence = -1
        self.target = None
        self._attempts = 0
        self.target_sequence = None
        self.flow_seed = None
        self.loss_detail = None

    @property
    def active(self):
        return self.state in self.ACTIVE

    @property
    def tracking(self):
        return self.enabled and self.state in {'DETECTADO', 'SIGUIENDO', 'CENTRANDO', 'ACERCANDO', 'ESTABILIZANDO', 'LISTO PARA CAPTURAR'}

    @property
    def status(self):
        display = {'VIGILANCIA': 'BUSCANDO', 'PERDIDO': 'OBJETIVO PERDIDO',
                   'CENTRANDO': 'SIGUIENDO · CENTRANDO'}.get(self.state, self.state)
        return f'{display} · {self.reason}'

    def enable(self):
        if self.physical and not self.directions_verified:
            self.state, self.reason = 'ERROR', 'PTZ físico bloqueado: desplazamiento y sentidos sin calibrar'
            self.enabled = False
            return False
        if self.active:
            return False
        self.enabled = True
        self.gate.invalidate()
        self._sequence = -1
        self.state, self.reason = 'VIGILANCIA', 'Buscando precinto'
        return True

    def abort(self, reason, *, lost=False):
        was_active = self.active
        self.decision = None
        self.target_sequence = None
        if lost and self.observe_only:
            if self.loss_detail is None:
                self.loss_detail = dict(previous_id=self.target.get('track_id'),
                    previous_source=self.target.get('source'), candidate_id=None,
                    candidate_source=None, iou=None, frame_age_ms=round((self.clock()-self._seen)*1000))
        if not self.observe_only and (self.enabled or self.active):
            self.stop()  # Existing public PTZ cancellation + stop barrier.
        self._token = None
        if was_active:
            self.gate.finished()
        self.gate.invalidate()
        self.state, self.reason = ('PERDIDO' if lost else 'ERROR'), reason
        self._terminal_at = self.clock()
        if not lost:
            self.enabled = False

    def disable(self, reason='Desactivado por operador'):
        self.abort(reason)
        self.enabled = False

    def _begin(self, packet):
        target = self.gate.confirmed_target
        if not target:
            return False
        self.target, self.shape = dict(target), packet['shape']
        self._remember_seed(packet)
        self.loss_detail = None
        self.target_sequence = packet['sequence']
        self._last_yolo = packet['submitted_at']
        self._seen = packet['submitted_at']
        self._started = self.clock()
        self._ready_at = self._started
        self._min_stamp = self._started - self.target_age_s
        self._moves = self._zooms = self._attempts = 0
        self._zoom_time = 0
        self._stable = None
        self._token = None
        self.capture_id = None
        self._waiting_after_stop = False
        self._rebind = None
        self._missing_since = None
        self._require_detector = False
        self._detector_missed = False
        self.state, self.reason = 'DETECTADO', 'Objetivo confirmado; preparando centrado'
        return True

    def observe(self, packet):
        if not self.enabled:
            return
        now = self.clock()
        if (packet.get('generation'), packet.get('stream_epoch')) != self.gate.context:
            return
        stamp = packet.get('submitted_at', float('-inf'))
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or not 0 <= now - stamp <= self.target_age_s:
            if self.tracking:
                self.abort('Coordenadas caducadas', lost=True)
            return
        seq = packet.get('sequence', -1)
        if seq <= self._sequence:
            return
        self._sequence = seq
        if self.state == 'VIGILANCIA':
            if packet.get('flow_only'):
                return
            if not self.available():
                self.gate.reason = 'Esperando video, motor OCR o fin de captura manual'
            self.gate.observe(packet, lambda: self._begin(packet), available=self.available())
            return
        if not self.tracking:
            return
        self.target_sequence = None
        if self.observe_only:
            self.decision = None
        if packet.get('shape') != self.shape:
            self.abort('Cambió la geometría de video', lost=True)
            return
        if self._waiting_after_stop and stamp <= self._min_stamp:
            self.reason = 'Esperando un fotograma tomado después del STOP confirmado'
            return
        candidates = []
        closest = None
        closest_overlap = -1.
        flow_only = packet.get('flow_only', False)
        for d in packet.get('detections', []):
            if not isinstance(d, dict) or d.get('source') not in ('yolo', 'auto-flow'):
                continue
            b = d.get('box', [])
            if len(b) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in b):
                continue
            iou = overlap(b, self.target['box'])
            if iou > closest_overlap:
                closest, closest_overlap = d, iou
            same_id = d.get('track_id') == self.target.get('track_id')
            if same_id and self.observe_only and d['source'] == 'yolo':
                confidence = d.get('confidence', 0)
                floor = self.gate.confidence
                if stamp - self._seen <= self.retain_s and (
                    (self.target['source'] == 'auto-flow' and self._missing_since is None)
                    or any(flow.get('source') == 'auto-flow' and flow.get('track_id') == d.get('track_id')
                           and overlap(flow['box'], b) >= .45 for flow in packet['detections'])
                ):
                    floor = min(floor, self.keep_confidence)
                if not isinstance(confidence, (int, float)) or not floor <= confidence <= 1:
                    continue
            if not same_id and (not self.observe_only or d['source'] != 'yolo'):
                continue
            if not same_id:
                confidence = d.get('confidence')
                if not isinstance(confidence, (int, float)) or not self.gate.confidence <= confidence <= 1:
                    continue
            if d.get('source') == 'auto-flow' and stamp - self._last_yolo > (self.flow_max_s if self.observe_only else 1.0):
                continue
            if (str(d.get('class_name', '')).lower() != 'precinto' or len(b) != 4
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in b)):
                continue
            h, w = self.shape[:2]
            if not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h) or covers_frame(b, w, h):
                continue
            ratio = (b[2] - b[0]) * (b[3] - b[1]) / max(1, (self.target['box'][2] - self.target['box'][0]) * (self.target['box'][3] - self.target['box'][1]))
            # ID alone is insufficient after zoom/restart: require geometric continuity.
            if .25 <= ratio <= 4 and iou >= (.10 if same_id else .45):
                candidates.append(d)
        if self.observe_only:
            self.loss_detail = dict(previous_id=self.target.get('track_id'),
                previous_source=self.target.get('source'),
                candidate_id=closest.get('track_id') if closest is not None else None,
                candidate_source=closest.get('source') if closest is not None else None,
                candidate_confidence=closest.get('confidence') if closest is not None else None,
                iou=round(closest_overlap, 3) if closest is not None else None,
                elapsed_ms=packet.get('elapsed_ms'),
                detector_age_ms=round((now-self._last_yolo)*1000),
                flow_reason=packet.get('flow_reason'),
                frame_age_ms=round((now-stamp)*1000), sequence=seq)
            # A unique detector match supersedes its predicted KLT box. Competing
            # detector boxes pause tracking instead of switching by confidence.
            direct = [d for d in candidates if d['source'] == 'yolo']
            if self._require_detector and not direct:
                return
            if not flow_only:
                self._detector_missed = not direct
            candidates = direct or candidates
            if len(candidates) == 1 and candidates[0].get('track_id') != self.target.get('track_id'):
                match = candidates[0]
                previous = self._rebind
                self._rebind = (match.get('track_id'), match['box'])
                if not previous or previous[0] != self._rebind[0] or overlap(previous[1], match['box']) < .5:
                    self._hold_observation('Cambio de ID pendiente de confirmar', now)
                    return
            if not flow_only:
                self._rebind = None
            if len(candidates) != 1:
                if direct:
                    self._require_detector = True
                reason = ('Asociación ambigua' if candidates else packet.get('flow_reason')
                          or 'Sin coincidencia fiable (ID, posición, escala o confianza)')
                self._hold_observation(reason, now)
                return
            self._missing_since = None
            if not flow_only:
                self._require_detector = False
                self.loss_detail = None
        if len(candidates) != 1:
            self.abort('Objetivo perdido; parada solicitada', lost=True)
            return
        best = max(candidates, key=lambda d: overlap(d['box'], self.target['box']))
        previous = self.target['box']
        self.target, self._seen = dict(best), stamp
        self.target_sequence = seq
        self._waiting_after_stop = False
        if best['source'] == 'yolo':
            self._last_yolo = stamp
            self._detector_missed = False
            self._remember_seed(packet)
        if self.observe_only:
            return
        if self.state in ('ESTABILIZANDO', 'LISTO PARA CAPTURAR'):
            q = best.get('sharpness', 0)
            ex, ey, size = self.geometry()
            good = (self._focus_verified and stamp >= self._min_stamp and now >= self._ready_at and not self.focus_pending()
                    and abs(ex) <= self.deadband and abs(ey) <= self.deadband and size >= self.target_size
                    and isinstance(q, (int, float)) and math.isfinite(q) and q >= self.sharpness
                    and overlap(previous, best['box']) >= .70)
            if not good:
                if (not self.focus_pending() and stamp >= self._min_stamp
                        and (abs(ex) > self.deadband or abs(ey) > self.deadband or size < self.target_size)):
                    self.state, self.reason = 'DETECTADO', 'Reajustando encuadre'
                else:
                    self.state = 'ESTABILIZANDO'
                self._stable = None
                return
            if self._stable is None:
                self._stable = (now, 1)
            else:
                self._stable = (self._stable[0], self._stable[1] + 1)
            if now - self._stable[0] >= self.settle_s and self._stable[1] >= 3 and self.available():
                self.state, self.reason = 'LISTO PARA CAPTURAR', 'Centrado, tamaño, estabilidad y nitidez verificados'

    def _remember_seed(self, packet):
        self.flow_seed = dict(sequence=packet['sequence'], box=list(self.target['box']),
                              track_id=self.target.get('track_id'), confidence=self.target['confidence'],
                              max_age_s=self.flow_max_s, appearance_min=self.appearance_min)

    @property
    def recovering(self):
        return self.observe_only and self.tracking and (
            self._missing_since is not None or self._detector_missed
            or self.clock() - self._last_yolo >= self.recheck_s)

    def visual_target(self):
        if not self.tracking or self.target is None:
            return []
        age = self.clock() - self._seen
        if self._missing_since is None and age <= self.FRESH_VISUAL_S:
            return [self.target]
        if age <= (self.retain_s if self.observe_only else self.RETAIN_S):
            return [dict(self.target, source='retained')]
        return []

    def _hold_observation(self, reason, now):
        self._missing_since = now if self._missing_since is None else self._missing_since
        self.decision = None
        self.state, self.reason = 'SIGUIENDO', reason

    def geometry(self):
        h, w = self.shape[:2]
        x1, y1, x2, y2 = self.target['box']
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        error_x = (center_x - w / 2) / (w / 2)
        error_y = (center_y - h / 2) / (h / 2)
        return (error_x, error_y,
                max((x2 - x1) / w, (y2 - y1) / h))

    def tick(self):
        if not self.enabled:
            return
        now = self.clock()
        if self.state in {'PERDIDO', 'FINALIZADO'}:
            if now - self._terminal_at >= 1:
                self.state, self.reason = 'VIGILANCIA', 'Buscando precinto'
            return
        if not self.active:
            return
        if self.state in {'CAPTURANDO', 'OCR'}:
            if now - self._phase_at > (30 if self.state == 'CAPTURANDO' else 90):
                self.abort('Tiempo de captura/OCR agotado; no se inicia otra captura')
            return
        if now - self._seen > self.target_age_s:
            reason = self.reason if self.observe_only and self._missing_since is not None else 'Sin seguimiento reciente'
            self.abort(reason, lost=True)
            return
        if self.target.get('source') == 'auto-flow' and now - self._last_yolo > (self.flow_max_s if self.observe_only else 1.0):
            self.abort('Seguimiento óptico caducado', lost=True)
            return
        if self.observe_only:
            if self.target['source'] == 'auto-flow' and self._missing_since is None:
                self.decision = None
                self.state, self.reason = 'SIGUIENDO', 'KLT provisional · esperando confirmación YOLO'
                return
            if self._missing_since is not None or now - self._seen > self.FRESH_VISUAL_S:
                self.decision = None
                return
            ex, ey, size = self.geometry()
            pan = self.pan_sign * (self.speed if ex > 0 else -self.speed) if abs(ex) > self.deadband else 0
            tilt = self.tilt_sign * (self.speed if ey > 0 else -self.speed) if abs(ey) > self.deadband else 0
            zoom = self.zoom_speed if not (pan or tilt) and size < self.target_size else 0
            self.decision = dict(pan=pan, tilt=tilt, zoom=zoom, duration_s=self.pulse_s)
            self.state = 'SIGUIENDO'
            self.reason = 'YOLO / ByteTrack'
            return
        if now - self._started > self.max_session_s:
            self.abort('Límite de tiempo de aproximación agotado')
            return
        if self._token is not None:
            self.reason = f'Esperando STOP del pulso {self._token}'
            if now - self._pulse_at > 4:
                self.abort('Parada del pulso PTZ no confirmada')
            return
        if self._waiting_after_stop or now < self._ready_at or self.focus_pending() or self._seen < self._min_stamp:
            self.reason = 'Esperando estabilización, enfoque u observación posterior al STOP'
            return
        if self.state == 'LISTO PARA CAPTURAR':
            if self.target.get('source') == 'auto-flow' and now - self._last_yolo > 1.0:
                self.abort('Seguimiento óptico caducado', lost=True)
                return
            if self.available() and self.capture_allowed():
                self.state, self.reason = 'CAPTURANDO', 'Captura 101 solicitada'
                self._attempts += 1
                self._phase_at = now
                self.capture_id = self.capture()
                if self.capture_id is None:
                    self.abort('La captura existente no aceptó la solicitud')
            return
        if self.state == 'ESTABILIZANDO':
            return
        ex, ey, size = self.geometry()
        if abs(ex) > self.deadband or abs(ey) > self.deadband:
            if self._moves >= self.max_moves:
                self.abort('Límite de intentos de centrado')
                return
            self.state, self.reason = 'CENTRANDO', f'Error normalizado x={ex:.3f}, y={ey:.3f}; zona muerta ±{self.deadband:.3f}'
            self._moves += 1
            cmd = dict(pan=self.pan_sign * (self.speed if ex > 0 else -self.speed) if abs(ex) > self.deadband else 0,
                       tilt=self.tilt_sign * (self.speed if ey > 0 else -self.speed) if abs(ey) > self.deadband else 0)
        elif size < self.target_size:
            if self._zooms >= self.max_zoom_steps or self._zoom_time + self.pulse_s > self.max_zoom_s:
                self.abort(f'Límite de zoom relativo: tamaño {size:.4f} < {self.target_size:.4f}; '
                           f'pasos {self._zooms}/{self.max_zoom_steps}; '
                           f'tiempo {self._zoom_time:.3f}/{self.max_zoom_s:.3f} s; '
                           'se solicita STOP sin capturar')
                return
            self.state, self.reason = 'ACERCANDO', 'Pulso limitado de zoom'
            self._zooms += 1
            self._zoom_time += self.pulse_s
            cmd = dict(zoom=self.zoom_speed)
        else:
            self._stabilize()
            return
        self._pulse_at = now
        self._token = self.pulse(duration_s=self.pulse_s, **cmd)
        if self._token is None:
            self.abort('PTZ no aceptó el pulso')

    def motion_result(self, packet):
        if not self.enabled or self._token is None or packet.get('token') != self._token:
            return
        if not packet.get('ok'):
            self.abort('PTZ rechazó movimiento/parada')
        elif packet.get('phase') == 'stopped':
            self._token = None
            self._waiting_after_stop = True
            self._min_stamp = self.clock()
            self._ready_at = self.clock() + .4

    def _stabilize(self):
        self._focus_verified = False
        self.state, self.reason = 'ESTABILIZANDO', 'Esperando enfoque, estabilidad y nitidez'
        self._stable = None
        self._min_stamp = self.clock()
        self._ready_at = self.clock() + self.settle_s
        if not self.focus_pending():
            self.focus()  # Existing public AUTO request; never edits its parameters.

    def focus_result(self, message):
        if self.enabled and self.state == 'ESTABILIZANDO':
            if message.startswith('Modo AUTO verificado'):
                self._focus_verified = True
                self._min_stamp = self.clock()
            else:
                self.abort('La cámara no confirmó AUTO')

    def photo_checked(self, capture_id, quality):
        if not self.enabled or self.state != 'CAPTURANDO' or capture_id != self.capture_id:
            return False
        if not quality:
            self.failed_capture(capture_id, 'Fotografía 101 sin nitidez/calidad suficiente')
            return False
        self.state, self.reason = 'OCR', 'Fotografía validada; OCR y guardado existentes'
        self._phase_at = self.clock()
        return True

    def failed_capture(self, capture_id, reason):
        if capture_id != self.capture_id or self.state not in {'CAPTURANDO', 'OCR'}:
            return
        self.capture_id = None
        if self.enabled and self._attempts <= self.retries:
            self._started = self.clock()
            self._seen = self._started  # Grace for a fresh observation; never use the old box for capture.
            self._stabilize()
        else:
            self.abort(reason)

    def ocr_result(self, capture_id, *, success):
        if capture_id != self.capture_id or self.state != 'OCR':
            return
        if not success:
            self.failed_capture(capture_id, 'OCR/guardado fallido; reintentos agotados')
            return
        self.state, self.reason = 'FINALIZADO', 'Resultado guardado; vigilancia sin duplicados'
        self.capture_id = None
        self.gate.finished(success=True)
        self._terminal_at = self.clock()
