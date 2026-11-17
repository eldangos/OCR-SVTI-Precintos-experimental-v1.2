"""Control ISAPI serializado con prioridad manual y espera de autoenfoque."""

import threading
import time
from collections import deque

from PySide6.QtCore import QThread, Signal

from config import AUTOFOCO_ESPERA_S


STOP = dict(pan=0, tilt=0, zoom=0, focus=0)


class PtzWorker(QThread):
    error = Signal(str)
    focus_status = Signal(str)
    configuration_status = Signal(str)

    def __init__(self, parent=None, *, focus_settle_s=AUTOFOCO_ESPERA_S):
        super().__init__(parent)

        self._focus_settle_s = max(0.0, float(focus_settle_s))
        self._lock = threading.Lock()
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
                self._barriers.append(
                    (self._service, self._ip, True, False)
                )

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
                )

        self._evento.set()

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
            self._auto = None
            self._intent += 1
            self._zoom_active = False
            self._manual = True
            self._last_was_stop = False

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

                    command = (
                        self._comando if barrier is None else None
                    )

                    if command is not None:
                        self._comando = None

                    service, ip = self._service, self._ip

                    auto = (
                        self._auto
                        if not barrier and not command
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
                        )
                        else None
                    )

                    if settings:
                        self._settings = None

                    active = self._activo

                if barrier:
                    target, address, retire, request_focus = barrier
                    ok = False

                    for _ in range(3):
                        ok = target.mover_camara(address, **STOP)

                        if ok:
                            break

                    self._report(target, ok)

                    with self._lock:
                        if not ok:
                            self._stop_faults.add(id(target))
                            self._last_was_stop = False
                            self._comando = self._auto = None

                        else:
                            self._stop_faults.discard(id(target))

                            if (
                                request_focus
                                == (self._generation, self._intent)
                                and target is self._service
                                and not self._manual
                                and self._comando is None
                                and not self._barriers
                            ):
                                self._auto = (
                                    self._generation,
                                    self._intent,
                                    time.monotonic()
                                    + self._focus_settle_s,
                                )

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
                    with self._lock:
                        valid = (
                            command[:2]
                            == (self._generation, self._intent)
                            and not self._barriers
                            and self._activo
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

                        ok = service.mover_camara(
                            ip,
                            **command[2],
                        )

                        self._report(service, ok)

                        if not ok:
                            self.enviar()

                    continue

                if auto and service:
                    def valid():
                        with self._lock:
                            return (
                                self._activo
                                and auto[:2]
                                == (self._generation, self._intent)
                                and not self._manual
                                and not self._barriers
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