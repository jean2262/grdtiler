import logging

from shapely import Polygon, MultiPolygon

logger = logging.getLogger(__name__)


def crosses_antimeridian(polygon: Polygon | MultiPolygon, threshold: float = 150) -> bool:
    """
    Return True if *polygon* crosses the antimeridian (±180° longitude).

    Detects crossing by inspecting consecutive vertex pairs: if the absolute
    longitude difference exceeds 180° the edge must wrap around the antimeridian.
    ``MultiPolygon`` geometries are first merged into a single polygon.

    Args:
        polygon (Polygon or MultiPolygon): Shapely geometry to test.
        threshold (float, optional): Minimum absolute longitude (degrees) that
            both vertices must exceed before the crossing check is applied.
            Defaults to 150.

    Returns:
        bool: True if the polygon crosses the antimeridian, False otherwise.
    """
    if isinstance(polygon, MultiPolygon):
        polygon = polygon.geoms[0].union(polygon.geoms[1])
    coords = list(polygon.exterior.coords)

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]

        if (lon1 > threshold and lon2 < -threshold) or (lon1 < -threshold and lon2 > threshold):
            if abs(lon1 - lon2) > 180:
                return True
    return False


def normalize_longitudes_to_360(polygon: Polygon) -> Polygon | None:
    """
    Convert all negative longitudes in *polygon* to the [0, 360] range.

    Useful when a footprint spans the antimeridian and standard [-180, 180]
    coordinates produce a degenerate geometry. Returns ``None`` on failure so
    callers can handle the error without an exception.

    Args:
        polygon (Polygon): Shapely polygon with coordinates in [-180, 180].

    Returns:
        Polygon or None: New polygon with longitudes in [0, 360], or ``None``
        if the conversion raised an exception.
    """
    try:
        normalized_coords = [
            ((lon + 360) if lon < 0 else lon, lat)
            for lon, lat in polygon.exterior.coords
        ]
        return Polygon(normalized_coords)
    except Exception as e:
        logger.error(f"Error normalizing longitudes: {e}")
        return None
