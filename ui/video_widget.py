import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from config import ALTO_RETICULA_PCT, ANCHO_RETICULA_PCT, GROSOR_RETICULA


class VideoWidget(QWidget):
    roi_selected = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._regions = []
        self._yolo_detections = []
        self._frame_size = None
        self._selection_enabled = False
        self._selection_start = None
        self._selection_current = None
        self._placeholder = "Configurá la cámara para ver el video"
        self.setMinimumSize(720, 405)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_placeholder(self, texto: str):
        self._placeholder = texto
        self.update()

    def mostrar_frame(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        alto, ancho, canales = rgb.shape
        self._frame_size = (ancho, alto)
        imagen = QImage(
            rgb.data, ancho, alto, canales * ancho, QImage.Format.Format_RGB888
        ).copy()
        self._pixmap = QPixmap.fromImage(imagen)
        self.update()

    def limpiar(self):
        self._pixmap = None
        self._regions = []
        self._yolo_detections = []
        self._frame_size = None
        self.update()

    def mostrar_regiones(self, regions):
        self._regions = regions
        self.update()

    def mostrar_detecciones_yolo(self, detections):
        if self._yolo_detections == detections:
            return
        self._yolo_detections = detections
        self.update()

    def habilitar_seleccion_roi(self, enabled: bool):
        self._selection_enabled = bool(enabled)
        self._selection_start = None
        self._selection_current = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def tiene_imagen(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    def _rect_destino(self) -> QRect | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        vw, vh = self.width(), self.height()
        if pw <= 0 or ph <= 0 or vw <= 0 or vh <= 0:
            return None
        escala = min(vw / pw, vh / ph)
        dw, dh = int(pw * escala), int(ph * escala)
        x = (vw - dw) // 2
        y = (vh - dh) // 2
        return QRect(x, y, dw, dh)

    def _image_point(self, point):
        dest = self._rect_destino()
        if dest is None or self._frame_size is None:
            return None
        x = min(max(point.x(), dest.left()), dest.right())
        y = min(max(point.y(), dest.top()), dest.bottom())
        image_width, image_height = self._frame_size
        return (
            (x - dest.left()) * image_width / max(1, dest.width()),
            (y - dest.top()) * image_height / max(1, dest.height()),
        )

    def mousePressEvent(self, event):
        if self._selection_enabled and event.button() == Qt.MouseButton.LeftButton:
            dest = self._rect_destino()
            if dest is not None and dest.contains(event.position().toPoint()):
                point = event.position().toPoint()
                self._selection_start = point
                self._selection_current = point
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selection_start is not None:
            self._selection_current = event.position().toPoint()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._selection_start is not None and event.button() == Qt.MouseButton.LeftButton:
            start = self._selection_start
            end = event.position().toPoint()
            first = self._image_point(start)
            second = self._image_point(end)
            self._selection_start = None
            self._selection_current = None
            if first is not None and second is not None:
                x1, y1 = first
                x2, y2 = second
                if abs(x2 - x1) >= 8 and abs(y2 - y1) >= 8:
                    self.roi_selected.emit(
                        [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    )
            self.update()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#11111B"))
        dest = self._rect_destino()
        if dest is None:
            painter.setPen(QColor("#A6ADC8"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        painter.drawPixmap(dest, self._pixmap)
        rw = max(8, int(dest.width() * ANCHO_RETICULA_PCT))
        rh = max(8, int(dest.height() * ALTO_RETICULA_PCT))
        rx = dest.center().x() - rw // 2
        ry = dest.center().y() - rh // 2
        pen = QPen(QColor('#8893A0'), 1, Qt.PenStyle.DotLine)
        pen.setWidth(GROSOR_RETICULA)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rx, ry, rw, rh)
        painter.drawText(rx + 4, ry + 14, 'Guía fija')
        for region in self._regions:
            x1, y1, x2, y2 = region['box']
            sx, sy = dest.width()/self._pixmap.width(), dest.height()/self._pixmap.height()
            x, y = int(dest.x()+x1*sx), int(dest.y()+y1*sy)
            painter.setPen(QPen(QColor('#F9E2AF'), 2, Qt.PenStyle.DashLine))
            painter.drawRect(x, y, int((x2-x1)*sx), int((y2-y1)*sy))
            painter.drawText(x, max(dest.y()+12,y-4), 'Texto candidato' + (' estable' if region.get('stable') else ''))

        scale_x = dest.width() / self._pixmap.width()
        scale_y = dest.height() / self._pixmap.height()
        for detection in self._yolo_detections:
            x1, y1, x2, y2 = detection['box']
            x = int(dest.x() + x1 * scale_x)
            y = int(dest.y() + y1 * scale_y)
            width = max(1, int((x2 - x1) * scale_x))
            height = max(1, int((y2 - y1) * scale_y))
            source = detection.get('source')
            color, style = {
                'retained': ('#B0B8C4', Qt.PenStyle.DotLine),
                'auto-flow': ('#F9E2AF', Qt.PenStyle.DashLine),
                'manual-flow': ('#89B4FA', Qt.PenStyle.DashLine),
            }.get(source, ('#A6E3A1', Qt.PenStyle.SolidLine))
            painter.setPen(QPen(QColor(color), 2, style))
            painter.drawRect(x, y, width, height)
            confidence = detection.get('confidence')
            confidence_text = f"{confidence:.2f}" if confidence is not None else "manual"
            track_id = detection.get('track_id')
            track_text = f"ID {track_id}" if track_id is not None else "sin ID"
            origin = {'auto-flow': 'KLT', 'manual-flow': 'ROI manual',
                      'retained': 'Posición retenida'}.get(source, 'YOLO')
            label = f"{origin} · {detection.get('class_name', 'Precinto')} {confidence_text} · {track_text}"
            painter.drawText(x, max(dest.y() + 14, y - 4), label)

        if self._selection_start is not None and self._selection_current is not None:
            selection = QRect(self._selection_start, self._selection_current).normalized()
            painter.setPen(QPen(QColor('#89B4FA'), 2, Qt.PenStyle.DashLine))
            painter.drawRect(selection)
