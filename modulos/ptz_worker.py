"""Control ISAPI serializado con prioridad manual y espera de autoenfoque."""

import threading
import time
from collections import deque

from PySide6.QtCore import QThread, Signal

from config import AUTOFOCO_ESPERA_S


STOP = dict(pan=0, tilt=0, zoom=0, focus=0)
# Existing movement transport timeout, independent of the nominal pulse duration.
PULSE_RESPONSE_DEADLINE_S = .5


class PtzWorker(QThread):
    error = Signal(str)
    focus_status = Signal(str)
    configuration_status = Signal(str)
    pulse_result = Signal(object)
    pulse_diagnostic = Signal(object)

    def __init__(self, parent=None, *, focus_settle_s=AUTOFOCO_ESPERA_S):
        super().__init__(parent)

        self._focus_settle_s = max(0.0, float(focus_settle_s))
        self._lock = threading.RLock()
        self._pulse_timer = None
        self._pulse_watchdog = None
        self._pulse_token = None
        self._pulse_sequence = 0
        self._pulse_stop_service = None
        self._evento = threading.Event()
        self._barriers = deque()
        self._comando = None
        self._activo = True
        self._service = self._ip = None
        self._generation = self._intent = 0
        self._zoom_active = self._manual = False
        self._auto = None
        self._auto_running = False
        self._motion_kind = ()
        self._last_was_stop = False
        self._stop_faults = set()
        self._settings = None
        self._cleanup_threads = []

    def configurar(self, service, ip):
        with self._lock:
            if self._service is not None:
                if self._pulse_token is not None:
                    self._stop_pulse_locked(self._pulse_token, 'Desconexión', retire=True)
                elif not self._motion_kind and (
                    self._pulse_stop_service is self._service or any(
                        len(barrier) == 5 and barrier[0] is self._service for barrier in self._barriers
                    )
                ):
                    self._barriers.append((self._service, self._ip, True, False, None))
                else:
                    self._barriers.append((self._service, self._ip, True, False))

            self._generation += 1
            self._intent += 1
            self._service, self._ip = service, ip
            self._comando = None
            self._zoom_active = self._manual = False
            self._motion_kind = ()
            self._last_was_stop = False
            self._settings = None

            self._auto = (
                (self._generation, self._intent, time.monotonic())
                if service
                else None
            )

        self._evento.set()

    def enviar(self, pan=0, tilt=0, zoom=0, focus=0):
        command = dict(
            pan=pan,
            tilt=tilt,
            zoom=zoom,
            focus=focus,
        )

        with self._lock:
            if not self._activo or self._service is None:
                return
            stop = not any(command.values())
            if self._pulse_token is not None:
                self._stop_pulse_locked(self._pulse_token, 'Orden posterior del operador')
                if stop:
                    return

            if stop and self._last_was_stop:
                return

            self._last_was_stop = stop
            self._intent += 1
            self._auto = None

            if focus:
                self._manual = True
                self._zoom_active = False

            if stop:
                focus_request = (
                    (self._generation, self._intent)
                    if self._zoom_active
                    else None
                )

                barrier = (
                    self._service,
                    self._ip,
                    False,
                    focus_request,
                )

                if not self._barriers or self._barriers[-1] != barrier:
                    self._barriers.append(barrier)

                self._comando = None
                self._zoom_active = False
                self._motion_kind = ()

            else:
                kind = tuple(
                    key for key, value in command.items() if value
                )

                if (
                    self._motion_kind
                    and self._motion_kind != kind
                ) or id(self._service) in self._stop_faults:
                    self._barriers.append(
                        (self._service, self._ip, False, False)
                    )

                self._motion_kind = kind

                if zoom:
                    self._zoom_active = True
                    self._manual = False

                self._comando = (
                    self._generation,
                    self._intent,
                    command,
                    None,
                    time.monotonic(),
                )
            token = (self._generation, self._intent)

        self._evento.set()
        return token

    def enviar_pulso(self, *, pan=0, tilt=0, zoom=0, duration_s=.15):
        """Bounded automatic command; its stop timer does not depend on the UI."""
        if (not .05 <= duration_s <= .4 or max(abs(pan), abs(tilt), abs(zoom)) > 30
                or not any((pan, tilt, zoom)) or (zoom and (pan or tilt))):
            raise ValueError('Pulso PTZ fuera de los límites de seguridad')
        command = dict(pan=pan, tilt=tilt, zoom=zoom, focus=0)
        with self._lock:
            token = None
            if self._activo and self._service is not None and not self._motion_kind:
                if self._pulse_token is not None:
                    self._stop_pulse_locked(self._pulse_token, 'Pulso posterior')
                if id(self._service) in self._stop_faults and not self._barriers:
                    self._barriers.append((self._service, self._ip, False, False, None))
                self._pulse_sequence += 1
                # Identidad propia para que el pulso no invalide las solicitudes de AUTO.
                token = (self._generation, -self._pulse_sequence)
                self._pulse_token = token
                self._comando = (*token, command, duration_s, time.monotonic())
                self._evento.set()
        self._pulse_event(token, 'queued' if token else 'rejected',
                          command=command, duration_s=duration_s,
                          reason=None if token else 'Worker inactivo, servicio no configurado o control manual activo')
        return token

    def _pulse_event(self, token, phase, **details):
        self.pulse_diagnostic.emit(dict(token=token, phase=phase,
                                        monotonic=time.monotonic(), **details))

    def _move_safely(self, service, address, command, token=None, *, pulse_stop=False):
        started = time.monotonic()
        if token is not None:
            self._pulse_event(token, 'service_call', command=command)
        try:
            if pulse_stop:
                ok = bool(service.detener_ptz(address))
            else:
                ok = bool(service.mover_camara(address, **command))
            detail = getattr(service, 'ultimo_error', None) if not ok else None
        except Exception as exc:
            ok, detail = False, f'Excepción del servicio: {type(exc).__name__}'
        if token is not None:
            self._pulse_event(token, 'service_response', command=command, ok=ok,
                              elapsed_s=time.monotonic()-started, detail=detail)
        if not ok:
            self.error.emit(detail or 'Movimiento/parada no confirmado por el servicio ISAPI')
        return ok

    def _stop_pulse(self, generation, intent, reason='Temporizador nominal agotado'):
        with self._lock:
            if (generation, intent) == self._pulse_token and self._activo:
                self._stop_pulse_locked((generation, intent), reason)
            else:
                self._pulse_event((generation, intent), 'timer_cancelled', reason='Orden posterior o desconexión')

    def _stop_pulse_locked(self, token, reason, *, retire=False):
        self._pulse_token = None
        for timer in (self._pulse_timer, self._pulse_watchdog):
            if timer is not None:
                timer.cancel()
        self._pulse_timer = self._pulse_watchdog = None
        if self._comando is not None and self._comando[:2] == token:
            self._comando = None
        self._barriers.append((self._service, self._ip, retire, False, token))
        self._evento.set()
        self._pulse_event(token, 'stop_requested', reason=reason)

    def enfoque_pendiente(self):
        with self._lock:
            return bool(
                self._auto
                or self._auto_running
                or any(barrier[3] for barrier in self._barriers)
            )

    def solicitar_autofoco(self):
        """Solicita parada antes de programar el autoenfoque."""
        with self._lock:
            if not self._activo or self._service is None:
                return
            if self._pulse_token is not None:
                self._stop_pulse_locked(self._pulse_token, 'Solicitud de autoenfoque')

            self._intent += 1
            self._auto = self._comando = None
            self._manual = self._zoom_active = False
            self._motion_kind = ()
            self._last_was_stop = True

            self._barriers.append(
                (
                    self._service,
                    self._ip,
                    False,
                    (self._generation, self._intent),
                )
            )

        self._evento.set()

    def cancelar_automatismos(self):
        with self._lock:
            if self._pulse_token is not None:
                self._stop_pulse_locked(self._pulse_token, 'Cancelación del pulso')
            elif self._motion_kind:
                self.enviar()

    def cambiar_dia_noche(self, modo):
        with self._lock:
            self._settings = (self._generation, modo)

        self._evento.set()

    def detener_hilo(self):
        self.configurar(None, None)

        with self._lock:
            self._activo = False

        self._evento.set()

    def _report(self, service, ok):
        if not ok:
            self.error.emit(
                service.ultimo_error
                or "La cámara no confirmó la orden ISAPI"
            )

    def run(self):
        wait_timeout = None

        while True:
            self._evento.wait(wait_timeout)
            self._evento.clear()
            wait_timeout = None

            while True:
                with self._lock:
                    barrier = (
                        self._barriers.popleft()
                        if self._barriers
                        else None
                    )
                    if barrier and len(barrier) == 5:
                        self._pulse_stop_service = barrier[0]

                    command = (
                        self._comando if barrier is None else None
                    )

                    if command is not None:
                        self._comando = None

                    service, ip = self._service, self._ip

                    auto = (
                        self._auto
                        if not barrier and not command and self._pulse_token is None
                        else None
                    )

                    if auto and auto[2] > time.monotonic():
                        wait_timeout = max(
                            0.0,
                            auto[2] - time.monotonic(),
                        )
                        auto = None

                    elif auto:
                        self._auto = None
                        self._auto_running = True

                    settings = (
                        self._settings
                        if (
                            not barrier
                            and not command
                            and not auto
                            and not self._motion_kind
                            and self._pulse_token is None
                        )
                        else None
                    )

                    if settings:
                        self._settings = None

                    active = self._activo

                if barrier:
                    target, address, retire, request_focus = barrier[:4]
                    pulse_stop = len(barrier) == 5
                    token = barrier[4] if pulse_stop else None
                    ok = False

                    for _ in range(3):
                        ok = self._move_safely(target, address, STOP, token, pulse_stop=pulse_stop)

                        if ok:
                            break

                    with self._lock:
                        if pulse_stop:
                            self._pulse_stop_service = None
                        if not ok:
                            self._stop_faults.add(id(target))
                            self._last_was_stop = False
                            if not pulse_stop:
                                self._comando = self._auto = None
                                if target is self._service and self._pulse_token is not None:
                                    self._stop_pulse_locked(self._pulse_token, 'STOP previo fallido')

                        else:
                            self._stop_faults.discard(id(target))

                            if (
                                request_focus
                                == (self._generation, self._intent)
                                and target is self._service
                                and not self._manual
                                and (self._comando is None or self._comando[3] is not None)
                                and all(len(pending) == 5 for pending in self._barriers)
                            ):
                                self._auto = (
                                    self._generation,
                                    self._intent,
                                    time.monotonic()
                                    + self._focus_settle_s,
                                )

                    if token is not None:
                        self.pulse_result.emit(dict(token=token, phase='stopped', ok=ok))

                    if retire:
                        cleanup = threading.Thread(
                            target=target.cerrar,
                            daemon=True,
                        )

                        self._cleanup_threads = [
                            thread
                            for thread in self._cleanup_threads
                            if thread.is_alive()
                        ]

                        self._cleanup_threads.append(cleanup)
                        cleanup.start()

                    continue

                if command:
                    pulsed = command[3] is not None
                    with self._lock:
                        valid = (
                            command[:2]
                            == (self._pulse_token if pulsed else (self._generation, self._intent))
                            and not self._barriers
                            and self._activo
                            and (not pulsed or id(service) not in self._stop_faults)
                        )

                    if valid:
                        if (
                            command[2]["focus"]
                            and hasattr(service, "establecer_manual")
                        ):
                            def current():
                                with self._lock:
                                    return (
                                        command[:2]
                                        == (
                                            self._generation,
                                            self._intent,
                                        )
                                        and self._activo
                                    )

                            manual_ok = service.establecer_manual(
                                ip,
                                current,
                            )

                            if not manual_ok:
                                focus_state = getattr(
                                    service,
                                    "focus_capability_state",
                                    None,
                                )
                                focus_validation = getattr(
                                    service,
                                    "focus_validation",
                                    None,
                                )

                                if (
                                    focus_state
                                    in ("unknown", "incompatible")
                                    and focus_validation
                                ):
                                    message = focus_validation

                                else:
                                    message = (
                                        service.ultimo_error
                                        or focus_validation
                                        or (
                                            "Modo MANUAL no confirmado; "
                                            "se conserva el mando manual original"
                                        )
                                    )

                                self.focus_status.emit(message)

                            if not current():
                                continue

                        if command[3] is not None:
                            with self._lock:
                                if (command[:2] != self._pulse_token
                                        or time.monotonic() - command[4] > .3):
                                    self._pulse_event(command[:2], 'discarded',
                                        age_s=time.monotonic()-command[4],
                                        reason='Orden sustituida o más de 300 ms en cola antes del envío')
                                    self._stop_pulse(*command[:2], reason='Pulso descartado antes del envío')
                                    self.pulse_result.emit(dict(token=command[:2], phase='started', ok=False))
                                    continue
                                self._pulse_watchdog = threading.Timer(
                                    PULSE_RESPONSE_DEADLINE_S, self._stop_pulse,
                                    args=(*command[:2], 'Plazo de seguridad del transporte agotado'))
                                self._pulse_watchdog.daemon = True
                                self._pulse_event(command[:2], 'safety_timer_started',
                                                  deadline_s=PULSE_RESPONSE_DEADLINE_S)
                                self._pulse_watchdog.start()
                        ok = self._move_safely(service, ip, command[2],
                                               command[:2] if command[3] is not None else None)

                        if len(command) > 3 and command[3] is not None:
                            with self._lock:
                                current = (command[:2] == self._pulse_token
                                           and self._activo and not self._barriers)
                                if ok and current:
                                    self._pulse_timer = threading.Timer(command[3], self._stop_pulse,
                                                                         args=command[:2])
                                    self._pulse_timer.daemon = True
                                    self._pulse_event(command[:2], 'pulse_timer_started',
                                                      duration_s=command[3], reference='service_response')
                                    self._pulse_timer.start()
                                else:
                                    self._pulse_event(command[:2], 'pulse_timer_not_started',
                                        reason='Respuesta rechazada o pulso cancelado/agotado durante la petición')
                                if not ok and current:
                                    self._stop_pulse(*command[:2], reason='Fallo o timeout del servicio')
                            self.pulse_result.emit(dict(token=command[:2], phase='started', ok=ok))

                        if not ok:
                            if command[3] is None:
                                self.enviar()

                    elif command[3] is not None:
                        self._pulse_event(command[:2], 'discarded', reason='Orden invalidada antes del envío')
                        self._stop_pulse(*command[:2], reason='Pulso invalidado')
                        self.pulse_result.emit(dict(token=command[:2], phase='started', ok=False))

                    continue

                if auto and service:
                    def valid():
                        with self._lock:
                            return (
                                self._activo
                                and auto[:2]
                                == (self._generation, self._intent)
                                and not self._manual
                                and all(len(pending) == 5 for pending in self._barriers)
                            )

                    try:
                        if valid() and hasattr(service, "enfocar_nativo"):
                            status = service.enfocar_nativo(ip, valid)

                            if valid() and isinstance(status, str):
                                self.focus_status.emit(status)

                    except Exception as exc:
                        if valid():
                            self.focus_status.emit(
                                "Autofoco no verificado: "
                                f"{type(exc).__name__}"
                            )

                    finally:
                        with self._lock:
                            self._auto_running = False

                    continue

                if settings and service:
                    with self._lock:
                        valid_setting = (
                            settings[0] == self._generation
                            and not self._barriers
                            and self._activo
                            and self._comando is None
                            and not self._motion_kind
                        )

                    if valid_setting:
                        ok = service.configurar_modo_dia_noche(
                            ip,
                            settings[1],
                        )

                        self.configuration_status.emit(
                            "Modo cámara: " + settings[1]
                            if ok
                            else (
                                service.ultimo_error
                                or "Modo cámara rechazado"
                            )
                        )

                    continue

                if not active:
                    for cleanup in self._cleanup_threads:
                        cleanup.join()

                    return

                break
