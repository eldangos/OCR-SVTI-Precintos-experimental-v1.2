"""Integración Qt offscreen sin red, vídeo ni modelos nativos."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QLineEdit

from modulos.excel_manifest import RegistroEsperado
from modulos.ocr_analysis import AnalysisResult, Candidate, Detection
from modulos.camera_service import CameraService
from modulos.ocr_worker import OcrWorker
from modulos.ptz_worker import PtzWorker
from modulos.reading_log import ReadingLogStore
from ui.main_window import MainWindow
from motor import MotorWindow
from modulos.ocr_reader import RecognitionResult
from PySide6.QtWidgets import QTableWidgetItem
from config import CANAL_RTSP_FOTO, CANAL_RTSP_VIDEO


class UiSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._registro_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._registro_tmp.cleanup)
        self.patches = [patch.object(OcrWorker,'start'), patch.object(PtzWorker,'start'),
                        patch.object(MainWindow,'_inicio'),
                        patch(
                            'ui.main_window.ReadingLogStore',
                            side_effect=lambda: ReadingLogStore(
                                Path(self._registro_tmp.name) / 'registro.sqlite3'
                            ),
                        )]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.window = MainWindow()
        self.window._timer.stop()
        self.addCleanup(self.window.deleteLater)

    def _prepare_capture(self):
        w = self.window
        w._generation, w._video_epoch = 3, 2
        w._service = Mock()
        w._ocr_listo = True
        w._video.mostrar_frame(np.zeros((180, 320, 3), np.uint8))
        w._btn_capturar.setEnabled(True)
        return w

    def test_lost_video_clears_picture_overlay_and_capture(self):
        w = self.window
        w._video.mostrar_frame(np.zeros((200,300,3),np.uint8))
        w._ocr_listo = True
        w._video_epoch = 1
        w._service = Mock(ultimo_error='Sin señal')
        w._service.copiar_muestra.return_value = None
        w._tick_video()
        self.assertFalse(w._video.tiene_imagen())
        self.assertFalse(w._btn_capturar.isEnabled())
        self.assertIsNone(w._video_epoch)

    def test_manual_capture_passes_entire_photo_without_manifest(self):
        w = self._prepare_capture()
        frame = np.zeros((180, 320, 3), np.uint8)
        frame[165, 10] = 255
        captured_at = datetime(2026, 11, 10, 12, 30, tzinfo=timezone.utc)
        w._captura_id_activa = 1
        w._ocr_ocupado = True
        packet = dict(
            frame=frame,
            stream_epoch=2,
            capture_id=1,
            captured_at=captured_at,
        )
        with tempfile.TemporaryDirectory() as tmp, \
                patch('ui.main_window.asegurar_carpeta_capturas', return_value=Path(tmp)), \
                patch.object(w._ocr, 'pedir_ocr') as read:
            w._procesar_captura_alta(3, packet)
            np.testing.assert_array_equal(read.call_args.args[0],frame)
            self.assertEqual(read.call_args.kwargs['input_mode'],'full_photo')
            self.assertNotIn('manifest',read.call_args.kwargs)

    def test_capture_reserves_daily_names_and_preserves_received_timestamp(self):
        w = self._prepare_capture()
        frame = np.zeros((60, 80, 3), np.uint8)
        captured_at = datetime(2026, 11, 10, 12, 30, 45, 123456, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as tmp, \
                patch('ui.main_window.asegurar_carpeta_capturas', return_value=Path(tmp)), \
                patch.object(w._ocr, 'pedir_ocr') as read:
            for capture_id, expected in ((1, 'P0001'), (2, 'P0002')):
                w._captura_secuencia = capture_id
                w._captura_id_activa = capture_id
                w._ocr_ocupado = True
                w._procesar_captura_alta(
                    3,
                    dict(
                        frame=frame,
                        stream_epoch=2,
                        capture_id=capture_id,
                        captured_at=captured_at,
                    ),
                )
                context = read.call_args.kwargs['context']
                evidence = context['evidence']
                image = Path(evidence['full_image'])
                metadata = json.loads(Path(evidence['metadata']).read_text(encoding='utf-8'))

                self.assertEqual(image.name, expected + '.png')
                self.assertEqual(Path(context['report']).name, expected + '_ocr.json')
                self.assertEqual(metadata['capture_id'], expected)
                self.assertEqual(metadata['captured_at'], captured_at.isoformat(timespec='microseconds'))
                self.assertEqual(evidence['captured_at'], metadata['captured_at'])
                self.assertEqual(context['capture_id'], capture_id)
                self.assertTrue(w._finalizar_captura(capture_id))

            historical = Path(tmp) / '2026-11-10' / 'P0003.png'
            historical.write_bytes(b'historical image')
            w._captura_secuencia = 3
            w._captura_id_activa = 3
            w._ocr_ocupado = True
            w._procesar_captura_alta(
                3,
                dict(
                    frame=frame,
                    stream_epoch=2,
                    capture_id=3,
                    captured_at=captured_at,
                ),
            )
            image = Path(read.call_args.kwargs['context']['evidence']['full_image'])
            self.assertEqual(image.name, 'P0004.png')
            self.assertEqual(historical.read_bytes(), b'historical image')

    def test_ocr_worker_saves_full_png_and_input_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = np.arange(60 * 80 * 3, dtype=np.uint8).reshape((60, 80, 3))
            image = Path(tmp) / 'P0001.png'
            evidence = {'full_image': str(image)}
            context = {'evidence': evidence, 'save_input': True}
            result = RecognitionResult(
                'fixture',
                'full_photo',
                status='texto_reconocido',
                recognized_text='ZX-591874Q',
                suggested_text='ZX-591874Q',
            )
            worker = OcrWorker()
            worker.pedir_ocr(frame, input_mode='full_photo', context=context)
            analyzed = []

            def finish(packet):
                analyzed.append(packet)
                worker.detener()

            worker.analyzed.connect(finish, Qt.ConnectionType.DirectConnection)
            with patch('modulos.ocr_reader.recognize', return_value=result):
                worker.run()

            self.assertEqual(len(analyzed), 1)
            self.assertTrue(image.is_file())
            import cv2
            np.testing.assert_array_equal(cv2.imread(str(image)), frame)
            self.assertEqual(
                evidence['input_sha256'],
                hashlib.sha256(frame.tobytes()).hexdigest(),
            )

    def test_capture_thread_construction_and_start_errors_release_busy_state(self):
        for failure_point in ('construct', 'start'):
            with self.subTest(failure_point=failure_point):
                w = self._prepare_capture()
                thread = Mock()
                thread.start.side_effect = RuntimeError('fixture')
                thread.is_alive.return_value = False
                thread_factory = (
                    patch('ui.main_window.threading.Thread', side_effect=RuntimeError('fixture'))
                    if failure_point == 'construct'
                    else patch('ui.main_window.threading.Thread', return_value=thread)
                )
                with thread_factory, \
                        patch('ui.main_window.CameraService.url_rtsp', return_value='rtsp://fixture'), \
                        patch('ui.main_window.QMessageBox.warning') as warning:
                    w._capturar()

                self.assertFalse(w._ocr_ocupado)
                self.assertIsNone(w._captura_id_activa)
                self.assertIsNone(w._captura_thread)
                self.assertTrue(w._btn_capturar.isEnabled())
                warning.assert_called_once()

    def test_late_capture_and_ocr_results_cannot_release_new_capture(self):
        w = self._prepare_capture()
        w._captura_secuencia = 2
        w._captura_id_activa = 2
        w._ocr_ocupado = True
        old_context = dict(generation=3, stream_epoch=2, capture_id=1)
        with patch('ui.main_window.QMessageBox.warning') as warning, \
                patch.object(w._ocr, 'pedir_ocr') as read:
            w._on_captura_alta_error(3, 2, 1, 'late capture error')
            w._procesar_captura_alta(
                3,
                dict(frame=None, stream_epoch=2, capture_id=1),
            )
            w._on_ocr_failed_context(dict(context=old_context, error='late OCR error'))
            w._on_ocr_analisis(dict(context=old_context, analysis=Mock()))

            self.assertTrue(w._ocr_ocupado)
            self.assertEqual(w._captura_id_activa, 2)
            warning.assert_not_called()
            read.assert_not_called()

            w._procesar_captura_alta(
                3,
                dict(frame=None, stream_epoch=1, capture_id=2),
            )
            self.assertFalse(w._ocr_ocupado)
            self.assertIsNone(w._captura_id_activa)

    def test_video_connects_on_low_resolution_channel(self):
        w = self.window
        with patch('ui.main_window.CameraService') as camera:
            camera.return_value.iniciar_lectura.return_value = Mock()
            w._conectar()
            camera.url_rtsp.assert_called_once_with(
                w._cfg.usuario, w._cfg.password, w._cfg.ip, CANAL_RTSP_VIDEO
            )
            w._desconectar()

    def test_capture_always_uses_photo_channel_in_background(self):
        w = self.window
        w._canal_activo,w._video_epoch = CANAL_RTSP_VIDEO,1
        w._service = Mock()
        w._btn_capturar.setEnabled(True)
        with patch('ui.main_window.CameraService.url_rtsp', return_value='rtsp://fixture') as url, \
                patch('ui.main_window.threading.Thread') as thread:
            w._capturar()
            url.assert_called_once_with(
                w._cfg.usuario, w._cfg.password, w._cfg.ip, CANAL_RTSP_FOTO
            )
            thread.assert_called_once()
            thread.return_value.start.assert_called_once()
            w._service.copiar_muestra.assert_not_called()

    def test_capture_worker_logs_rtsp_exception_and_emits_recovery_signal(self):
        w = self.window
        errors = []
        w._captura_alta_error.connect(lambda *args: errors.append(args))
        with patch.object(
            CameraService,
            'capturar_frame_rtsp',
            side_effect=RuntimeError('simulated FFmpeg read failure'),
        ) as capture, self.assertLogs('ui.main_window', level='ERROR') as logs:
            w._capturar_alta_res('rtsp://fixture/101', 3, 2, 9)

        capture.assert_called_once_with(
            'rtsp://fixture/101',
            open_timeout_ms=5000,
            read_timeout_ms=5000,
        )
        self.assertEqual(
            errors,
            [(3, 2, 9, 'No se pudo capturar desde el canal 101')],
        )
        self.assertIn(
            'High-resolution RTSP capture failed before returning a frame',
            '\n'.join(logs.output),
        )
        self.assertIn('simulated FFmpeg read failure', '\n'.join(logs.output))

    def test_video_default_never_queues_detection_and_skips_repeated_conversion(self):
        w = self.window
        w._ocr_listo,w._video_epoch = True,1
        w._service = Mock()
        w._service.copiar_muestra.side_effect = [(np.zeros((180,320,3),np.uint8),1,time.monotonic(),1),
                                                (None,1,time.monotonic(),1)]
        with patch.object(w._video,'mostrar_frame') as display,patch.object(w._ocr,'pedir_video') as infer:
            w._tick_video()
            w._tick_video()
            self.assertEqual(display.call_count,1)
            infer.assert_not_called()

    def test_yolo_uses_existing_video_samples_without_opening_another_rtsp_stream(self):
        w = self.window
        frame = np.zeros((180, 320, 3), np.uint8)
        captured_at = time.monotonic()
        w._generation = 5
        w._video_epoch = 2
        w._last_display_sequence = -1
        w._service = Mock(ultimo_error=None)
        w._service.copiar_muestra.return_value = (frame, 24, captured_at, 2)
        w._yolo_enabled = True

        with patch.object(w._yolo_worker, 'submit_frame') as submit, \
                patch.object(CameraService, 'iniciar_lectura') as open_stream:
            w._tick_video()

        submit.assert_called_once_with(
            frame,
            sequence=24,
            submitted_at=captured_at,
            generation=5,
            stream_epoch=2,
        )
        open_stream.assert_not_called()

    def test_both_applications_display_same_full_visual_observation(self):
        w = self.window
        w._generation,w._video_epoch = 3,2
        w._service = Mock()
        text = 'UNKNOWN ZX-591874Q EXTRA'
        result = RecognitionResult('fixture','full_photo',status='texto_reconocido',recognized_text=text,
                                   suggested_text=text,confidence=.96,selected_variant='BGR')
        m = MotorWindow()
        self.addCleanup(m.deleteLater)
        m._imagenes,m._indice_actual,m._procesando = [Path('fixture.png')],0,True
        m._inicio_imagen = time.perf_counter()
        m._tabla.setRowCount(1)
        for column in range(7):
            m._tabla.setItem(0,column,QTableWidgetItem(''))
        with tempfile.TemporaryDirectory() as tmp,patch('ui.main_window.asegurar_carpeta_capturas',return_value=Path(tmp)):
            w._on_ocr_analisis(dict(analysis=result,context=dict(generation=3,stream_epoch=2,evidence={},report=str(Path(tmp)/'ocr.json'))))
        m._on_analisis_detallado(dict(analysis=result,context=None))
        m._on_ocr_resultado(text,.96,'BGR',.96)
        self.assertEqual(w._txt_ocr.toPlainText(),m._tabla.item(0,1).text())
        self.assertEqual(w._txt_ocr.toPlainText(),text)
        self.assertIn('sin manifiesto',w._lbl_comparacion.text())

    def test_motor_export_keeps_visual_evidence_beside_excel(self):
        m = MotorWindow()
        self.addCleanup(m.deleteLater)
        m._imagenes,m._indice_actual,m._procesando = [Path('fixture.png')],0,True
        m._inicio_imagen = time.perf_counter()
        m._tabla.setRowCount(1)
        for column in range(7):
            m._tabla.setItem(0,column,QTableWidgetItem(''))
        text = 'UNKNOWN ZX-591874Q EXTRA'
        result = RecognitionResult('fixture','full_photo',status='texto_reconocido',recognized_text=text,
                                   suggested_text=text,raw_observations=[dict(text=text)],confidence=.96)
        m._on_analisis_detallado(dict(analysis=result,context=None))
        m._on_ocr_resultado(text,.96,'BGR',.96)
        with tempfile.TemporaryDirectory() as tmp,patch('motor.QFileDialog.getSaveFileName',return_value=(str(Path(tmp)/'export.xlsx'),'')),patch('motor.QMessageBox.information'):
            m._exportar_excel()
            self.assertTrue((Path(tmp)/'export.xlsx').exists())
            files = list(Path(tmp).glob('export_evidence_*.json'))
            self.assertEqual(len(files),1)
            saved = json.loads(files[0].read_text(encoding='utf8'))
            self.assertEqual(saved['records'][0]['analysis']['recognized_text'],text)
            self.assertEqual(saved['records'][0]['analysis']['raw_observations'][0]['text'],text)

    def test_stale_detection_and_old_capture_are_ignored(self):
        w = self.window
        w._generation,w._video_epoch = 3,2
        w._service = Mock()
        for gen,epoch,age in ((2,2,0),(3,1,0),(3,2,9)):
            w._on_regiones(dict(generation=gen,stream_epoch=epoch,captured_at=time.monotonic()-age,regions=[]))
            self.assertIsNone(w._region_packet)
        with patch.object(w._ocr,'pedir_ocr') as infer:
            w._procesar_captura_alta(2,dict(frame=None,stream_epoch=2))
            w._procesar_captura_alta(3,dict(frame=None,stream_epoch=1))
            infer.assert_not_called()

    def test_key_release_stops_even_after_focus_moves_to_input(self):
        w = self.window
        w._service = Mock()
        w._held_ptz.add(Qt.Key.Key_J)
        event = QKeyEvent(QEvent.Type.KeyRelease,Qt.Key.Key_J,Qt.KeyboardModifier.NoModifier)
        with patch.object(w,'_atajos_globales_activos',return_value=False),patch.object(w._ptz,'enviar') as send:
            self.assertTrue(w.eventFilter(QLineEdit(),event))
            send.assert_called_once_with(0,0,0,0)

    def test_held_direction_keys_combine_and_keep_remaining_direction(self):
        w = self.window
        w._service = Mock()
        widget = QLineEdit()
        key_press = lambda key: QKeyEvent(
            QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier
        )
        key_release = lambda key: QKeyEvent(
            QEvent.Type.KeyRelease, key, Qt.KeyboardModifier.NoModifier
        )

        with patch.object(w, '_atajos_globales_activos', return_value=True), \
                patch.object(w._ptz, 'enviar') as send:
            self.assertTrue(w.eventFilter(widget, key_press(Qt.Key.Key_W)))
            self.assertEqual(send.call_args.kwargs['tilt'], 50)

            send.reset_mock()
            self.assertTrue(w.eventFilter(widget, key_press(Qt.Key.Key_D)))
            self.assertEqual(send.call_args.kwargs['pan'], 50)
            self.assertEqual(send.call_args.kwargs['tilt'], 50)

            send.reset_mock()
            self.assertTrue(w.eventFilter(widget, key_release(Qt.Key.Key_D)))
            self.assertEqual(send.call_args.kwargs['pan'], 0)
            self.assertEqual(send.call_args.kwargs['tilt'], 50)

            send.reset_mock()
            self.assertTrue(w.eventFilter(widget, key_release(Qt.Key.Key_W)))
            send.assert_called_once_with(0, 0, 0, 0)

    def test_rotation_slider_and_fixed_optic_speeds_reach_ptz_commands(self):
        w = self.window
        w._service = Mock()
        self.assertFalse(hasattr(w, '_slider_vel_zoom'))
        self.assertFalse(hasattr(w, '_slider_vel_foco'))
        w._slider_vel_mov.setValue(73)

        with patch.object(w._ptz, 'enviar') as send:
            self.assertTrue(w._comando_tecla(Qt.Key.Key_D))
            self.assertEqual(send.call_args.kwargs['pan'], 73)

            send.reset_mock()
            self.assertTrue(w._comando_tecla(Qt.Key.Key_I))
            self.assertEqual(send.call_args.kwargs['zoom'], 100)

            send.reset_mock()
            self.assertTrue(w._comando_tecla(Qt.Key.Key_J))
            self.assertEqual(send.call_args.kwargs['focus'], 10)

            send.reset_mock()
            self.assertTrue(w._comando_tecla(Qt.Key.Key_L))
            self.assertEqual(send.call_args.kwargs['focus'], -10)

            send.reset_mock()
            w._btn_right.motion_start.emit()
            self.assertEqual(send.call_args.kwargs['pan'], 73)

            send.reset_mock()
            w._btn_zoom_in.motion_start.emit()
            self.assertEqual(send.call_args.kwargs['zoom'], 100)

            send.reset_mock()
            w._btn_foco_cerca.motion_start.emit()
            self.assertEqual(send.call_args.kwargs['focus'], 10)

    def test_only_rotation_speed_remains_adjustable_and_optic_speeds_are_fixed(self):
        w = self.window

        self.assertEqual((w._slider_vel_mov.minimum(), w._slider_vel_mov.maximum()), (1, 100))
        self.assertFalse(hasattr(w, '_slider_vel_zoom'))
        self.assertFalse(hasattr(w, '_slider_vel_foco'))
        self.assertEqual(w._vel_zoom, 100)
        self.assertEqual(w._vel_foco, 10)

        w._slider_vel_mov.setValue(0)
        self.assertEqual(w._slider_vel_mov.value(), 1)
        w._slider_vel_mov.setValue(101)
        self.assertEqual(w._slider_vel_mov.value(), 100)

    def test_audit_precedes_state_change_and_later_reading_never_downgrades(self):
        w = self.window
        w._manifiesto = [RegistroEsperado(precinto_esperado='B 0097312')]
        d = Detection('B 0097312','B 0097312',.98,[[1,1],[90,1],[90,20],[1,20]],'fixture',0,'fixture')
        c = Candidate('B 0097312',.98,.98,.98,2,True,[],[d])
        analysis = AnalysisResult('fixture','seal_roi',status='reconocido',confirmed_identifier=c.text,candidates=[c])
        w._evidence = {'fixture':True}
        with tempfile.TemporaryDirectory() as tmp,patch('ui.main_window.asegurar_carpeta_capturas',return_value=Path(tmp)):
            with patch('ui.main_window.append_decision',side_effect=OSError('fixture')):
                with self.assertRaises(OSError):
                    w._actualizar_manifiesto_por_ocr(analysis)
            self.assertEqual(w._manifiesto[0].estado,'Pendiente')
            w._actualizar_manifiesto_por_ocr(analysis)
            self.assertEqual(w._manifiesto[0].estado,'Llegado')
            analysis.status,analysis.confirmed_identifier = 'ambiguo',None
            w._actualizar_manifiesto_por_ocr(analysis)
            self.assertEqual(w._manifiesto[0].estado,'Llegado')
            events = [json.loads(s) for s in (Path(tmp)/'decisiones.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([e['accepted'] for e in events],[True,False])

    def test_close_keeps_running_workers_alive(self):
        w = self.window
        event = Mock()
        with patch.object(w._ocr,'isRunning',return_value=True),patch('ui.main_window.QTimer.singleShot') as later:
            w.closeEvent(event)
            event.ignore.assert_called_once()
            event.accept.assert_not_called()
            later.assert_called()

    def test_queued_error_from_previous_connection_does_not_affect_new_capture(self):
        w = self.window
        w._generation,w._video_epoch = 3,2
        w._service = Mock()
        w._ocr_ocupado = True
        with patch.object(w,'_on_ocr_error') as show_error:
            for generation,epoch in ((2,2),(3,1)):
                w._on_ocr_failed_context(dict(context=dict(generation=generation,stream_epoch=epoch),error='fixture'))
            show_error.assert_not_called()
        self.assertTrue(w._ocr_ocupado)

    def test_close_cancels_deferred_startup(self):
        self.window._closing = True
        with patch.object(self.window,'_abrir_config') as config,patch.object(self.window,'_conectar') as connect:
            # Llamar la implementación real: _inicio se sustituye en setUp.
            self.patches[2].stop()
            self.window._inicio()
            config.assert_not_called()
            connect.assert_not_called()

    def test_display_confidence_belongs_to_selected_identifier(self):
        w = self.window
        w._generation,w._video_epoch = 3,2
        w._service = Mock()
        w._manifiesto = [RegistroEsperado(precinto_esperado='B 0097312')]
        wrong = Candidate('0097312',.99,.99,.99,2,True,[],[])
        selected = Candidate('B 0097312',.94,.94,.94,2,True,[],[])
        analysis = AnalysisResult('fixture','seal_roi',status='reconocido',confirmed_identifier=selected.text,
                                   suggested_text=wrong.text,candidates=[wrong,selected])
        with tempfile.TemporaryDirectory() as tmp,patch('ui.main_window.asegurar_carpeta_capturas',return_value=Path(tmp)):
            packet = dict(analysis=analysis,context=dict(generation=3,stream_epoch=2,evidence={},report=str(Path(tmp)/'ocr.json')))
            w._on_ocr_analisis(packet)
        self.assertEqual(w._txt_ocr.toPlainText(),'B 0097312')
        self.assertIn('94.0',w._lbl_confianza.text())

    def test_ocr_result_is_added_to_persistent_reading_log(self):
        w = self.window
        w._generation,w._video_epoch = 3,2
        w._service = Mock()
        analysis = RecognitionResult(
            'fixture',
            'full_photo',
            status='texto_reconocido',
            recognized_text='ASDASD',
            suggested_text='ASDASD',
            confidence=.95,
            selection_score=.92,
            selected_variant='fixture',
        )
        with tempfile.TemporaryDirectory() as tmp, \
                patch('ui.main_window.asegurar_carpeta_capturas', return_value=Path(tmp)):
            photo = Path(tmp) / 'lectura_input.png'
            photo.write_bytes(b'fixture')
            report = Path(tmp) / 'lectura_ocr.json'
            packet = dict(
                analysis=analysis,
                context=dict(
                    generation=3,
                    stream_epoch=2,
                    evidence=dict(full_image=str(photo), input_sha256='fixture-sha'),
                    report=str(report),
                ),
            )
            w._on_ocr_analisis(packet)

        entries = w._registro_lecturas.listar_filas()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].first.codigo, 'ASDASD')
        self.assertEqual(entries[0].first.foto_path, str(photo))
        self.assertEqual(entries[0].first.input_sha256, 'fixture-sha')

    def test_ui_desechar_y_recuperar_mueve_la_relacion_completa(self):
        w = self.window
        first_id = w._registro_lecturas.agregar_lectura(codigo='ASDASD')
        second_id = w._registro_lecturas.agregar_lectura(codigo='ASJDJAD')
        relation_id = w._registro_lecturas.relacionar(first_id, second_id)
        w._refrescar_registro_lecturas()

        with patch.object(w, '_filas_seleccionadas', return_value=[0]):
            w._desechar_lecturas()
        self.assertEqual(w._tabla_lecturas.rowCount(), 0)
        self.assertEqual(w._tabla_desechadas.rowCount(), 1)
        self.assertEqual(w._registro_desechadas_filas[0].relation_id, relation_id)

        with patch.object(w, '_filas_seleccionadas', return_value=[0]):
            w._recuperar_desechadas()
        self.assertEqual(w._tabla_lecturas.rowCount(), 1)
        self.assertEqual(w._tabla_desechadas.rowCount(), 0)


if __name__=='__main__':
    unittest.main()
