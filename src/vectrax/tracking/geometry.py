"""Boxes in normalized image coordinates (0..1), center + size."""

from dataclasses import dataclass

__all__ = ["Box"]


@dataclass(frozen=True, slots=True)
class Box:
    cx: float
    cy: float
    w: float
    h: float

    @classmethod
    def from_xywh_px(cls, x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> "Box":
        return cls((x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h)

    def to_xywh_px(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        w = self.w * img_w
        h = self.h * img_h
        return round(self.cx * img_w - w / 2), round(self.cy * img_h - h / 2), round(w), round(h)
