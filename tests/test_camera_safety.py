"""Regresiones con transportes simulados; nunca accede a una cámara real."""
import threading
import time
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from modulos.camera_service import CameraService
from modulos.ptz_worker import PtzWorker


class CameraSafetyTests(unittest.TestCase):
    def service(self, status=200, body=""):
        service = CameraService("fixture", "fixture")
        service.session = Mock()
        response = Mock(status_code=status, text=body, content=body.encode())
        service.session.put.return_value = response
        return service

    def test_http_rejection_is_failure(self):
        service = self.service(403)
        self.assertFalse(service.mover_camara("fixture", pan=30))

    def test_isapi_rejection_with_http_200_is_failure(self):
        service = self.service(body='<ResponseStatus><statusCode>4</statusCode><statusString>Invalid Operation</statusString></ResponseStatus>')
        self.assertFalse(service.mover_camara("fixture", focus=5))

    def test_unexpected_http_200_body_cannot_confirm_control(self):
        service = self.service(body='<html>Login required</html>')
        self.assertFalse(service.mover_camara('fixture',pan=30))

    def test_failed_pan_stop_still_attempts_lens_stop(self):
        service = self.service()
        service.session.put.side_effect = [TimeoutError("fixture"), Mock(status_code=200, text="")]
        self.assertFalse(service.mover_camara("fixture"))
        self.assertEqual(service.session.put.call_count, 2)

    def test_disconnected_frame_cannot_be_used_as_live(self):
        service = self.service()
        service.ultimo_frame = np.zeros((20, 30, 3), np.uint8)
        service.conectado = False
        self.assertIsNone(service.copiar_frame())

    def test_old_decoded_frame_expires_even_if_connected_flag_is_true(self):
        service = self.service()
        service.ultimo_frame = np.zeros((20,30,3),np.uint8)
        service.conectado = True
        service._frame_at = time.monotonic()-9
        self.assertIsNone(service.copiar_frame())

    def test_rtsp_capture_logs_open_failure_without_reading_and_releases(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch.object(CameraService, '_abrir_rtsp', return_value=capture), \
            self.assertLogs('modulos.camera_service', level='WARNING') as logs:
            result = CameraService.capturar_frame_rtsp('rtsp://fixture/101')

        self.assertIsNone(result)
        capture.read.assert_not_called()
        capture.release.assert_called_once()
        output = '\n'.join(logs.output)
        self.assertIn('RTSP open failed', output)
        self.assertIn('RTSP VideoCapture constructor returned after', output)
        self.assertIn('RTSP isOpened() returned False in', output)

    def test_rtsp_capture_logs_read_failures_and_attempt_count(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch.object(CameraService, '_abrir_rtsp', return_value=capture), \
                self.assertLogs('modulos.camera_service', level='WARNING') as logs:
            result = CameraService.capturar_frame_rtsp('rtsp://fixture/101')

        self.assertIsNone(result)
        self.assertEqual(capture.read.call_count, 3)
        capture.release.assert_called_once()
        output = '\n'.join(logs.output)
        self.assertIn('RTSP VideoCapture constructor returned after', output)
        self.assertIn('RTSP isOpened() returned True in', output)
        self.assertIn('RTSP read returned no frame on attempt 1/3', output)
        self.assertIn('attempt 3/3 after', output)
        self.assertIn('RTSP read failed: no valid frame after 3 attempts', output)

    def test_persistent_rtsp_open_defaults_remain_two_seconds(self):
        with patch('modulos.camera_service.cv2.VideoCapture') as video_capture:
            CameraService._abrir_rtsp('rtsp://fixture/102')

        args = video_capture.call_args.args
        self.assertEqual(args[2][1], 2000)
        self.assertEqual(args[2][3], 2000)

    def test_capture_rtsp_open_can_override_timeouts_to_five_seconds(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch.object(CameraService, '_abrir_rtsp', return_value=capture) as open_stream:
            CameraService.capturar_frame_rtsp(
                'rtsp://fixture/101',
                open_timeout_ms=5000,
                read_timeout_ms=5000,
            )

        open_stream.assert_called_once_with(
            'rtsp://fixture/101',
            open_timeout_ms=5000,
            read_timeout_ms=5000,
        )

    def test_rtsp_capture_logs_first_frame_arriving_after_two_seconds(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((2, 3, 3), dtype=np.uint8))
        clock = [0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 2.51, 2.51]

        with patch.object(CameraService, '_abrir_rtsp', return_value=capture), \
                patch('modulos.camera_service.time.monotonic', side_effect=clock), \
                self.assertLogs('modulos.camera_service', level='WARNING') as logs:
            frame = CameraService.capturar_frame_rtsp(
                'rtsp://fixture/101',
                open_timeout_ms=5000,
                read_timeout_ms=5000,
            )

        self.assertEqual(frame.shape, (2, 3, 3))
        output = '\n'.join(logs.output)
        self.assertIn('RTSP first valid frame on read 1/3', output)
        self.assertIn('read duration 2.500 s', output)
        self.assertIn('total elapsed 2.510 s', output)
        capture.release.assert_called_once()

    def test_rtsp_capture_logs_open_exception(self):
        with patch.object(
            CameraService,
            '_abrir_rtsp',
            side_effect=RuntimeError('simulated open failure'),
        ), self.assertLogs('modulos.camera_service', level='ERROR') as logs:
            with self.assertRaisesRegex(RuntimeError, 'simulated open failure'):
                CameraService.capturar_frame_rtsp('rtsp://fixture/101')

        self.assertIn('RTSP VideoCapture constructor raised', '\n'.join(logs.output))

    def test_rtsp_capture_logs_read_exception_and_releases(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = RuntimeError('simulated FFmpeg read failure')
        with patch.object(CameraService, '_abrir_rtsp', return_value=capture), \
                self.assertLogs('modulos.camera_service', level='ERROR') as logs:
            with self.assertRaisesRegex(RuntimeError, 'simulated FFmpeg read failure'):
                CameraService.capturar_frame_rtsp('rtsp://fixture/101')

        capture.release.assert_called_once()
        self.assertIn(
            'RTSP read raised an exception on attempt 1/3',
            '\n'.join(logs.output),
        )

    def test_duplicate_sample_avoids_copy_without_bypassing_freshness(self):
        service = self.service()
        service.ultimo_frame = Mock()
        service.conectado = True
        service._frame_at = time.monotonic()
        service.frame_sequence = 22
        sample = service.copiar_muestra(after_sequence=22)
        self.assertIsNone(sample[0])
        service.ultimo_frame.copy.assert_not_called()
        service._frame_at = time.monotonic()-9
        self.assertIsNone(service.copiar_muestra(after_sequence=22))

    def test_stop_before_reader_start_never_reopens_stream(self):
        service = self.service()
        service.detener_lectura()
        with patch.object(service,'_abrir_rtsp') as open_stream:
            service.lector_rtsp('fixture')
            open_stream.assert_not_called()

    def test_failed_open_always_releases_capture(self):
        service = self.service()
        capture = Mock()
        capture.isOpened.return_value = False
        def opening(_):
            service._stop.set()
            return capture
        with patch.object(service,'_abrir_rtsp',side_effect=opening):
            service.lector_rtsp('fixture')
        capture.release.assert_called_once()
        self.assertIsNone(service.copiar_frame())

    def test_three_failed_stops_block_pending_movement(self):
        service = Mock(ultimo_error='fixture')
        stopped = threading.Event()
        def move(ip,**cmd):
            if not any(cmd.values()):
                if service.mover_camara.call_count >= 3:
                    stopped.set()
                return False
            return True
        service.mover_camara.side_effect = move
        worker = PtzWorker()
        worker.configurar(service,'fixture')
        worker.enviar(pan=10)
        worker.enviar()
        worker.enviar(pan=20)
        worker.start()
        try:
            self.assertTrue(stopped.wait(2))
        finally:
            worker.detener_hilo()
            self.assertTrue(worker.wait(3000))
        self.assertFalse(any(c.kwargs.get('pan') for c in service.mover_camara.call_args_list))

    def test_stop_is_a_barrier_not_a_replaceable_command(self):
        calls = []
        delivered = threading.Event()
        service = Mock(ultimo_error=None)

        def move(ip, **command):
            calls.append(command)
            if command["pan"] == 20:
                delivered.set()
            return True

        service.mover_camara.side_effect = move
        worker = PtzWorker()
        worker.configurar(service, "fixture")
        worker.enviar(pan=10)
        worker.enviar()
        worker.enviar(pan=20)
        worker.start()
        try:
            self.assertTrue(delivered.wait(2))
            self.assertGreaterEqual(len(calls), 2)
            self.assertTrue(all(value == 0 for value in calls[0].values()))
        finally:
            worker.detener_hilo()
            self.assertTrue(worker.wait(3000))

    def test_reconfigure_never_replays_old_movement_on_new_service(self):
        old, new = Mock(ultimo_error=None), Mock(ultimo_error=None)
        worker = PtzWorker()
        worker.configurar(old, "old")
        worker.enviar(focus=5)
        worker.configurar(new, "new")
        worker.start()
        time.sleep(.05)
        worker.detener_hilo()
        self.assertTrue(worker.wait(3000))
        self.assertFalse(any(c.kwargs.get("focus") == 5 for c in new.mover_camara.call_args_list))


if __name__ == "__main__":
    unittest.main()
