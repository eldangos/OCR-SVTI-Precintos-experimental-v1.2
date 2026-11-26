"""Exercise actual UI adapter methods with simulated capture and no Qt imports.

This deliberately does not claim to exercise Qt signals, RTSP or native OCR.
Run: python -B tests/test_yolo_vigilance_integration.py
"""
import ast
from pathlib import Path
from types import SimpleNamespace

from test_yolo_vigilance import Simulator


def adapter(sim):
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'ui/main_window.py').read_text(encoding='utf-8-sig'))
    main = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MainWindow')
    names = {'_on_yolo_result', '_sync_yolo_vigilance', '_disparar_captura_yolo', '_tick_video'}
    methods = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[ast.ClassDef(name='Adapter', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(time=SimpleNamespace(monotonic=lambda: sim.now), YOLO_RESULT_MAX_AGE_S=2,
                     DETECCION_EN_VIVO=False)
    exec(compile(module, '<actual UI vigilance methods>', 'exec'), namespace)
    w = namespace['Adapter']()
    w._vigilance = sim.gate
    w._auto = SimpleNamespace(enabled=False, tracking=False)
    w._btn_auto = SimpleNamespace(isChecked=lambda: False)
    w._yolo_enabled, w._closing, w._ocr_ocupado = True, False, False
    w._generation, w._video_epoch = 1, 2
    w._captura_id_activa = w._captura_thread = None
    w.messages, w.frames, w.requests = [], [], []
    w._lbl_worker = SimpleNamespace(setText=lambda text: None)
    w._lbl_yolo_estado = SimpleNamespace(setText=w.messages.append)
    w._video = SimpleNamespace(mostrar_detecciones_yolo=w.frames.append,
                               mostrar_frame=lambda f: None, mostrar_regiones=lambda r: None)
    w._btn_capturar = SimpleNamespace(isEnabled=lambda: not w._ocr_ocupado and not w._closing)
    w._actualizar_captura = lambda: None
    w._on_yolo_log = lambda message: None
    def capture():
        assert w._vigilance.processing
        w._ocr_ocupado = True
        w._captura_id_activa = len(w.requests) + 1
        w.requests.append(w._captura_id_activa)
    w._capturar = capture
    w._region_packet = None
    w._last_display_sequence = -1
    w._last_sequence = -1
    w._yolo_sample_saved = True
    w._yolo_last_result_at = None
    w._ocr_listo = True
    w.sent = []
    w._yolo_worker = SimpleNamespace(activity=lambda: 'Detector listo', accepts_frame=lambda: True, is_ready=lambda: True,
                                    submit_frame=lambda *args, **kw: w.sent.append(kw))
    w._set_estado = lambda *args: None
    return w


def packet(sim, sequence):
    return dict(sequence=sequence, generation=1, stream_epoch=2, submitted_at=sim.now,
                shape=[120, 160, 3], detections=[dict(box=[40, 30, 65, 80],
                class_name='Precinto', confidence=.9, source='yolo')])


def test_monitor_mode_displays_detections_without_capturing():
    s = Simulator()
    w = adapter(s)
    for seq in range(1, 15):
        s.now += .5
        w._on_yolo_result(packet(s, seq))
    assert not w.requests and not w._vigilance.processing and w.frames[-1]


def test_adapter_rejects_old_or_disabled_results():
    s = Simulator()
    w = adapter(s)
    for seq in range(1, 10):
        p = packet(s, seq)
        p['generation'] = 0
        w._on_yolo_result(p)
    w._yolo_enabled = False
    for seq in range(10, 20): w._on_yolo_result(packet(s, seq))
    assert not w.requests


def test_existing_manual_capture_is_also_latched():
    s = Simulator()
    w = adapter(s)
    w._captura_id_activa, w._ocr_ocupado = 9, True
    w._sync_yolo_vigilance()
    assert w._vigilance.processing and not w._vigilance.armed
    w._captura_id_activa, w._ocr_ocupado = None, False
    w._sync_yolo_vigilance()
    assert not w._vigilance.processing and not w._vigilance.armed


def test_busy_capture_thread_prevents_premature_completion():
    s = Simulator()
    w = adapter(s)
    s.gate.started()
    w._captura_thread = SimpleNamespace(is_alive=lambda: True)
    w._sync_yolo_vigilance()
    assert s.gate.processing


def test_actual_tick_throttles_frames_and_pauses_during_pipeline():
    s = Simulator(interval_s=.5)
    w = adapter(s)
    w._service = SimpleNamespace(copiar_muestra=lambda **kw: (object(), 1, s.now, 2))
    w._tick_video()
    for _ in range(10):
        s.now += .03
        w._tick_video()
    assert len(w.sent) == 1 and w.sent[0]['track_visual']
    s.now += .3
    w._tick_video()
    assert len(w.sent) == 2
    w._captura_id_activa, w._ocr_ocupado = 1, True
    s.now += 2
    w._tick_video()
    assert len(w.sent) == 2 and s.gate.processing


def test_worker_busy_drops_frames_without_building_queue():
    s = Simulator()
    w = adapter(s)
    w._service = SimpleNamespace(copiar_muestra=lambda **kw: (object(), 1, s.now, 2))
    w._yolo_worker.accepts_frame = lambda: False
    for _ in range(20):
        s.now += .1
        w._tick_video()
    assert not w.sent


def test_visual_tracking_can_bridge_first_missed_detection():
    s = Simulator()
    w = adapter(s)
    w._service = SimpleNamespace(copiar_muestra=lambda **kw: (object(), 1, s.now, 2))
    w._on_yolo_result(packet(s, 1))
    w._tick_video()
    assert w.sent[-1]['track_visual']
    s.gate.started()
    s.gate.finished()
    s.now += 1
    w._tick_video()
    assert w.sent[-1]['track_visual']


def test_child_skips_klt_entirely_in_search_mode():
    path = Path(__file__).resolve().parents[1] / 'ocr_tools/yolo_worker_process.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('overlap', 'follow_target')]
    def forbidden(*args):
        raise AssertionError('KLT must not run in search mode')
    namespace = dict(initialize_manual_tracker=forbidden, advance_manual_tracker=forbidden)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), namespace)
    frame = SimpleNamespace(shape=(120, 160, 3))
    detections = packet(Simulator(), 1)['detections']
    state, boxes, status = namespace['follow_target'](None, frame, detections, None,
                                                    now=100, track_visual=False)
    assert state is None and len(boxes) == 1 and status == 'Siguiendo'


if __name__ == '__main__':
    tests = [value for name, value in list(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print(f'{len(tests)} actual UI adapter checks passed with simulated capture')
