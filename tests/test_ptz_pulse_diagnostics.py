"""Targeted pulse diagnostics. All HTTP responses are mocked."""
import threading
import time
import unittest
from unittest.mock import Mock
import requests

from PySide6.QtCore import Qt
from modulos.camera_service import CameraService
from modulos.ptz_worker import PtzWorker


class PulseDiagnosticsTests(unittest.TestCase):
    def test_cancellation_during_auto_keeps_request_and_stops_only_ptz(self):
        service = CameraService('fixture', 'fixture')
        service.session = Mock()
        service.session.put.return_value = Mock(
            status_code=200, text='<ResponseStatus><statusCode>1</statusCode></ResponseStatus>')
        auto_started, resume_auto, stopped = (threading.Event() for _ in range(3))
        validity = []

        def autofocus(address, valid):
            auto_started.set()
            resume_auto.wait(2)
            validity.append(valid())
            return 'Modo AUTO verificado'

        def result(packet):
            if packet['phase'] == 'stopped' and packet['ok']:
                stopped.set()

        service.enfocar_nativo = Mock(side_effect=autofocus)
        worker = PtzWorker()
        worker.pulse_result.connect(result, Qt.ConnectionType.DirectConnection)
        worker.configurar(service, 'fixture.invalid')
        worker.start()
        try:
            self.assertTrue(auto_started.wait(2))
            self.assertIsNotNone(worker.enviar_pulso(pan=5))
            worker.cancelar_automatismos()
            resume_auto.set()
            self.assertTrue(stopped.wait(2))
            self.assertEqual(validity, [True])
            self.assertFalse(worker._manual)
            service.session.put.assert_called_once()
            call = service.session.put.call_args
            self.assertTrue(call.args[0].endswith('/ISAPI/PTZCtrl/channels/1/continuous'))
            self.assertEqual(call.kwargs['data'],
                             '<PTZData><pan>0</pan><tilt>0</tilt><zoom>0</zoom></PTZData>')
        finally:
            resume_auto.set()
            worker.detener_hilo()
            self.assertTrue(worker.wait(2000))

    def test_response_then_80ms_with_independent_safety_stop(self):
        for case in ('accepted', 'rejected', 'exception', 'timeout', 'slow_response',
                     'deadline_exceeded', 'expired_in_queue', 'operator_cancelled'):
            with self.subTest(case=case):
                service = CameraService('fixture', 'fixture')
                service.session = Mock()
                worker = PtzWorker()
                events, results = [], []
                done = threading.Event()
                def put(url, **kwargs):
                    data = kwargs['data']
                    moving = '<pan>5</pan>' in data
                    if moving and case == 'slow_response':
                        time.sleep(.12)  # Deliberately longer than unchanged 80 ms.
                    if moving and case == 'deadline_exceeded':
                        time.sleep(.60)  # Mock ignores the service timeout to exercise the independent guard.
                    if moving and case == 'timeout':
                        raise requests.ReadTimeout('fixture')
                    status = 403 if moving and case == 'rejected' else 200
                    return Mock(status_code=status, text='<ResponseStatus><statusCode>1</statusCode></ResponseStatus>')
                service.session.put.side_effect = put
                original = service.mover_camara
                def move(address, **cmd):
                    if case == 'exception' and cmd.get('pan') == 5:
                        raise RuntimeError('fixture')
                    return original(address, **cmd)
                service.mover_camara = move
                worker.pulse_diagnostic.connect(events.append, Qt.ConnectionType.DirectConnection)
                def result(packet):
                    results.append(packet)
                    if case == 'operator_cancelled' and packet['phase'] == 'started':
                        worker.cancelar_automatismos()
                    if packet['phase'] == 'stopped' or (case == 'expired_in_queue' and not packet['ok']):
                        done.set()
                worker.pulse_result.connect(result, Qt.ConnectionType.DirectConnection)
                worker.configurar(service, 'fixture.invalid')
                service.enfocar_nativo = Mock(return_value='Modo AUTO verificado')
                token = worker.enviar_pulso(pan=5, duration_s=.08)
                if case == 'expired_in_queue':
                    time.sleep(.33)
                worker.start()
                try:
                    self.assertTrue(done.wait(2))
                    selected = [e for e in events if e['token'] == token]
                    self.assertTrue(all(c.args[0].endswith('/ISAPI/PTZCtrl/channels/1/continuous')
                                        for c in service.session.put.call_args_list))
                    self.assertFalse(worker._manual)
                    if case == 'expired_in_queue':
                        self.assertTrue(any(e['phase'] == 'discarded' for e in selected))
                        self.assertFalse(any(e['phase'] == 'service_call' and e['command']['pan']
                                             for e in selected))
                        continue
                    calls = [e for e in selected if e['phase'] == 'service_call']
                    self.assertEqual(calls[0]['command'], dict(pan=5, tilt=0, zoom=0, focus=0))
                    self.assertEqual(calls[-1]['command'], dict(pan=0, tilt=0, zoom=0, focus=0))
                    responses = [e for e in selected if e['phase'] == 'service_response']
                    self.assertEqual(responses[0]['ok'], case not in ('rejected', 'exception', 'timeout'))
                    self.assertTrue(responses[-1]['ok'])
                    if case == 'rejected':
                        self.assertIn('HTTP 403', responses[0]['detail'])
                    if case == 'exception':
                        self.assertIn('RuntimeError', responses[0]['detail'])
                    stop_request = next(e for e in selected if e['phase'] == 'stop_requested')
                    timers = [e for e in selected if e['phase'] == 'pulse_timer_started']
                    if case in ('accepted', 'slow_response', 'operator_cancelled'):
                        self.assertEqual(len(timers), 1)
                        self.assertEqual(timers[0]['duration_s'], .08)
                        self.assertGreaterEqual(timers[0]['monotonic'], responses[0]['monotonic'])
                        if case == 'operator_cancelled':
                            self.assertIn('Cancelación', stop_request['reason'])
                        else:
                            self.assertGreaterEqual(stop_request['monotonic'] - timers[0]['monotonic'], .075)
                    else:
                        self.assertFalse(timers)
                    if case == 'deadline_exceeded':
                        self.assertLess(stop_request['monotonic'], responses[0]['monotonic'])
                        self.assertIn('seguridad', stop_request['reason'])
                    if case == 'timeout':
                        self.assertIn('ReadTimeout', responses[0]['detail'])
                    if case != 'exception':
                        movement = next(c for c in service.session.put.call_args_list if '<pan>5</pan>' in c.kwargs['data'])
                        self.assertTrue(movement.args[0].endswith('/ISAPI/PTZCtrl/channels/1/continuous'))
                    self.assertTrue(any(r['phase'] == 'stopped' and r['ok'] for r in results))
                finally:
                    worker.detener_hilo()
                    self.assertTrue(worker.wait(2000))
