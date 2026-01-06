from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ExifTags

from rag.models import Geo, NormalizedRecord
from rag.utils import parse_datetime


EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
GPS_TAGS = ExifTags.GPSTAGS


def _convert_ratio(value: Any) -> float | None:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _gps_to_decimal(gps_info: dict[int, Any]) -> Geo | None:
    if not gps_info:
        return None
    gps_data: dict[str, Any] = {}
    for key, val in gps_info.items():
        name = GPS_TAGS.get(key, key)
        gps_data[name] = val

    lat_vals = gps_data.get("GPSLatitude")
    lat_ref = gps_data.get("GPSLatitudeRef")
    lon_vals = gps_data.get("GPSLongitude")
    lon_ref = gps_data.get("GPSLongitudeRef")
    if not lat_vals or not lon_vals or not lat_ref or not lon_ref:
        return None

    def _to_deg(values: Any) -> float | None:
        if not isinstance(values, (list, tuple)) or len(values) < 3:
            return None
        deg = _convert_ratio(values[0])
        minutes = _convert_ratio(values[1])
        seconds = _convert_ratio(values[2])
        if deg is None or minutes is None or seconds is None:
            return None
        return deg + (minutes / 60.0) + (seconds / 3600.0)

    lat = _to_deg(lat_vals)
    lon = _to_deg(lon_vals)
    if lat is None or lon is None:
        return None
    if str(lat_ref).upper().startswith("S"):
        lat = -lat
    if str(lon_ref).upper().startswith("W"):
        lon = -lon
    return Geo(lat=lat, lon=lon)


def _exif_datetime(exif: dict[int, Any]) -> str | None:
    for key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
        tag = EXIF_TAGS.get(key)
        if tag and tag in exif:
            return exif[tag]
    return None


def _sidecar_json(path: Path) -> tuple[str | None, Geo | None]:
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamp = None
    geo = None
    if isinstance(payload, dict):
        photo_time = payload.get("photoTakenTime", {})
        timestamp = photo_time.get("timestamp") or photo_time.get("formatted")
        geo_data = payload.get("geoData") or payload.get("geoDataExif") or {}
        lat = geo_data.get("latitude")
        lon = geo_data.get("longitude")
        if lat is not None and lon is not None:
            try:
                geo = Geo(lat=float(lat), lon=float(lon))
            except (TypeError, ValueError):
                geo = None
    return timestamp, geo


def _iter_images(folder: Path) -> list[Path]:
    patterns = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff"}
    files: list[Path] = []
    for path in folder.rglob("*"):
        if path.suffix.lower() in patterns:
            files.append(path)
    return files


def load_photos(folder: Path) -> list[NormalizedRecord]:
    records: list[NormalizedRecord] = []
    for img_path in _iter_images(folder):
        exif = {}
        utc_dt = None
        geo = None
        try:
            with Image.open(img_path) as img:
                raw_exif = img._getexif() or {}
                exif = raw_exif
                exif_time = _exif_datetime(exif)
                utc_dt = parse_datetime(exif_time)
                gps_tag = EXIF_TAGS.get("GPSInfo")
                if gps_tag and gps_tag in exif:
                    geo = _gps_to_decimal(exif[gps_tag])
        except Exception:
            pass

        sidecar_ts, sidecar_geo = _sidecar_json(img_path.with_suffix(img_path.suffix + ".json"))
        if utc_dt is None and sidecar_ts:
            utc_dt = parse_datetime(sidecar_ts)
        if geo is None and sidecar_geo:
            geo = sidecar_geo

        if utc_dt is None:
            continue
        record_id = f"photo-{img_path.name}-{int(utc_dt.timestamp())}"
        records.append(
            NormalizedRecord(
                id=record_id,
                source="photo",
                utc_datetime=utc_dt,
                local_tz=None,
                text=f"Photo: {img_path.name}",
                geo=geo,
                participants=[],
                meta={"path": str(img_path)},
            )
        )
    return records
