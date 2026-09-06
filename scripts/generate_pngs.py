import sys
import os
import struct
import zlib
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.colors as mcolors
from PIL import Image
from omfiles import OmFileReader

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "output"
output_dir = sys.argv[2]      # z.B. "output/maps"
var_type = sys.argv[3]        # 't2m_mean', 'wind_mean', ...  (Pipeline ruft nur noch diese beiden auf)
# Optional: eigener Ordner fuer die *_spread .om-Dateien, falls die nicht im
# selben Ordner wie die Mean-Dateien liegen. Faellt auf data_dir zurueck,
# wenn nicht angegeben.
spread_data_dir = sys.argv[4] if len(sys.argv) > 4 else data_dir
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# var_type -> Substring, der im Dateinamen der zugehoerigen .om Datei steht
# ------------------------------
OM_FILENAME_PATTERNS = {
    "t2m_mean": "temperature_2m",
    "wind_mean": "wind_gusts_10m",
    "t2m_spread": "temperature_2m_spread",
    "wind_spread": "wind_gusts_10m_spread",
}

# ------------------------------
# Fuer welche Variablen die echten Werte zusaetzlich als DVAL-Chunk
# ins WebP eingebettet werden sollen (kein separates File noetig).
# ------------------------------
EMBED_DATA_VARS = {"t2m_mean", "wind_mean", "t2m_spread", "wind_spread"}

# ------------------------------
# Mean-Variable -> zugehoerige Spread-Variable. Fuer diese Mean-Typen wird
# zusaetzlich die passende *_spread .om-Datei geladen und deren Werte als
# zweiter, separater Chunk (SPREAD_CHUNK_FOURCC) in dieselbe .webp-Datei
# eingebettet. Es werden dadurch KEINE eigenen t2m_spread/wind_spread
# .webp-Bilder mehr erzeugt - die Pipeline soll dieses Skript nur noch mit
# var_type "t2m_mean" bzw. "wind_mean" aufrufen. Die *_spread .om-Dateien
# duerfen in einem eigenen Ordner liegen, siehe sys.argv[4] (spread_data_dir).
# ------------------------------
MEAN_TO_SPREAD_VAR = {
    "t2m_mean": "t2m_spread",
    "wind_mean": "wind_spread",
}
SPREAD_CHUNK_FOURCC = b"DSPR"

# ------------------------------
# Temperatur-Farben
# ------------------------------
t2m_bounds = list(range(-36, 50, 2))
t2m_colors = LinearSegmentedColormap.from_list(
    "t2m_smoooth",
    [
        "#F675F4", "#F428E9", "#B117B5", "#950CA2", "#640180",
        "#3E007F", "#00337E", "#005295", "#1292FF", "#49ACFF",
        "#8FCDFF", "#B4DBFF", "#B9ECDD", "#88D4AD", "#07A125",
        "#3FC107", "#9DE004", "#E7F700", "#F3CD0A", "#EE5505",
        "#C81904", "#AF0E14", "#620001", "#C87879", "#FACACA",
        "#E1E1E1", "#6D6D6D"
    ],
    N=len(t2m_bounds)
)
t2m_norm = BoundaryNorm(t2m_bounds, ncolors=len(t2m_bounds))

# ------------------------------
# Temperatur Spannweite-Farben
# ------------------------------
t2m_spread_bounds = [0.1, 0.2, 0.4, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10, 11, 12, 13, 14, 15, 16, 17, 18]
t2m_spread_colors = ListedColormap([
       "#1392FF", "#49ACFF", "#6DBCFF", "#91CCFF", "#C7E4FF", "#AAF682",
       "#B4F266", "#C8F04A", "#DCEC2E", "#EBE816", "#F4D90B", "#F4BD0B",
       "#F4A20B", "#F4880B", "#F46D0B", "#F4520B", "#E83709", "#CE2007",
       "#BA130F", "#A50B19", "#96010A", "#780000", "#8C3232", "#C87878",
       "#F0A0A0", "#FFC8C8", "#FFF0F0", "#C5C5C5", "#8A8A8A"
    ])
t2m_spread_norm = mcolors.BoundaryNorm(t2m_spread_bounds, t2m_spread_colors.N)

# ------------------------------
# Windböen-Farben
# ------------------------------
wind_bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 240, 260, 280, 300]
wind_colors = ListedColormap([
    "#68AD05", "#8DC00B", "#B1D415", "#D5E81C", "#FBFC22",
    "#FAD024", "#F9A427", "#FC7929", "#FB4D2B", "#EA2B57",
    "#FB22A5", "#FC22CE", "#FC22F5", "#FC62F8", "#FD80F8",
    "#FFBFFC", "#FEDFFE", "#FEFFFF", "#E1E0FF", "#C3C3FF",
    "#A5A5FF", "#A5A5FF", "#6868FE"
])
wind_norm = mcolors.BoundaryNorm(wind_bounds, wind_colors.N)


# ------------------------------
# Temperatur Spannweite-Farben
# ------------------------------
wind_spread_bounds = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90]
wind_spread_colors = ListedColormap([
       "#1392FF", "#49ACFF", "#6DBCFF", "#91CCFF", "#C7E4FF", "#AAF682",
       "#B4F266", "#C8F04A", "#DCEC2E", "#EBE816", "#F4D90B", "#F4BD0B",
       "#F4A20B", "#F4880B", "#F46D0B", "#F4520B", "#E83709", "#CE2007",
       "#BA130F", "#A50B19", "#96010A", "#780000", "#8C3232", "#C87878",
       "#F0A0A0", "#FFC8C8", "#FFF0F0", "#C5C5C5", "#8A8A8A"
    ])
wind_spread_norm = mcolors.BoundaryNorm(wind_spread_bounds, wind_spread_colors.N)

# ------------------------------
# Niederschlags-Farben 1h (tp)
# ------------------------------
prec_bounds = [0.0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 125]
prec_colors = ListedColormap([
    "#FFFFFF", "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4"
])
prec_norm = mcolors.BoundaryNorm(prec_bounds, prec_colors.N)

COLORMAPS = {
    "t2m_mean": (t2m_colors, t2m_norm),
    "wind_mean": (wind_colors, wind_norm),
    "t2m_spread": (t2m_spread_colors, t2m_spread_norm),
    "wind_spread": (wind_spread_colors, wind_spread_norm),
}

# Umrechnung Rohwert -> Anzeige-Einheit, je Variable
UNIT_CONVERT = {
    "t2m_mean": lambda v: v - 273.15 if np.nanmax(v) > 200 else v,  # K -> °C, nur falls noetig
    "wind_mean": lambda v: v * 3.6,  # m/s -> km/h
    "t2m_spread": lambda v: v,
    "wind_spread": lambda v: v * 3.6,  # m/s -> km/h (gleiche Einheit wie wind_mean)
}

# Quantisierungsschritt je Variable fuer den eingebetteten DVAL-Chunk
# (feiner als die Anzeige-Nachkommastellen, damit kein sichtbarer
# Genauigkeitsverlust entsteht).
QUANTUM_STEP = {
    "t2m_mean": 0.05,   # °C
    "wind_mean": 0.2,   # km/h
    "t2m_spread": 0.05,  # K
    "wind_spread": 0.05  # km/h
}
NAN_SENTINEL_I16 = -32768
DVAL_FOURCC = b"DVAL"

# ------------------------------
# Bounding Box (wie im GRIB2-Skript)
# ------------------------------
extent = [-3.94, 20.34, 43.18, 58.08]  # lon_min, lon_max, lat_min, lat_max

# Bounding Box fuer den eingebetteten DVAL-Chunk (nur t2m/wind) - hier
# reicht Deutschland + etwas Rand fuer Grenzregionen beim Hovern; das
# Farbbild selbst bleibt unveraendert auf der vollen Domaene.
GERMANY_BBOX_LONLAT = [5.5, 15.3, 47.0, 55.3]  # lon_min, lon_max, lat_min, lat_max

# ------------------------------
# EPSG:4326 -> EPSG:3857 (Web Mercator)
# ------------------------------
EARTH_RADIUS = 6378137.0
WEBMERCATOR_WIDTH = 1024

# ------------------------------
# ECMWF Reduced Gaussian Grid (Octahedral, "O<N>") - z.B. O320 fuer EC46 EPS.
# Die Datei speichert keine expliziten lat/lon-Arrays (siehe crs_wkt-Kind:
# "Reduced Gaussian Grid O320 (ECMWF)"), daher werden die Koordinaten hier
# aus der Gitterdefinition berechnet statt aus der Datei gelesen.
# ------------------------------
GAUSSIAN_N = 320  # aus crs_wkt-REMARK "Reduced Gaussian Grid O320" abgelesen


def build_octahedral_latlon(N=GAUSSIAN_N):
    L = N
    dy = 180.0 / (2 * L + 0.5)

    lats_full = []
    lons_full = []
    for y in range(2 * L):
        # nx pro Zeile: Nordhalbkugel (y < L) steigend 20,24,28,...;
        # Suedhalbkugel spiegelbildlich fallend - identisch zu
        # GridType.nxOf(y:) im Swift-Code.
        nx = (20 + y * 4) if y < L else ((2 * L - y - 1) * 4 + 20)
        lat = (L - y - 1) * dy + dy / 2
        lon = np.arange(nx) * (360.0 / nx)
        lon = np.where(lon >= 180, lon - 360, lon)  # wrap wie im Original
        lats_full.append(np.full(nx, lat))
        lons_full.append(lon)

    lat = np.concatenate(lats_full)
    lon = np.concatenate(lons_full)
    return lat, lon


def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y


def webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH):
    lon_min, lon_max, lat_min, lat_max = extent
    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)
    aspect = (y_max - y_min) / (x_max - x_min)
    out_height = max(int(round(out_width * aspect)), 1)
    x_new = np.linspace(x_min, x_max, out_width)
    y_new = np.linspace(y_min, y_max, out_height)  # aufsteigend: Süd -> Nord
    return x_new, y_new


def warp_o320_to_webmercator(data_flat, lat_flat, lon_flat, extent, method="linear", out_width=WEBMERCATOR_WIDTH):
    """data_flat/lat_flat/lon_flat: unregelmaessig verteilte Punkte
    (Reduced Gaussian Grid). Interpoliert per griddata auf das reguläre
    Web-Mercator-Zielraster. Gibt ein Array mit row0 = Sueden zurueck
    (wie das alte warp_equirect_to_webmercator)."""
    lon_min, lon_max, lat_min, lat_max = extent
    pad = 2.0  # Grad Rand, damit die Interpolation am Rand sauber bleibt
    mask = (
        (lat_flat >= lat_min - pad) & (lat_flat <= lat_max + pad) &
        (lon_flat >= lon_min - pad) & (lon_flat <= lon_max + pad)
    )
    pts = np.column_stack([lon_flat[mask], lat_flat[mask]])
    vals = data_flat[mask]

    x_new, y_new = webmercator_target_grid(extent, out_width=out_width)
    xx, yy = np.meshgrid(x_new, y_new)
    lon_grid = np.degrees(xx / EARTH_RADIUS)
    lat_grid = np.degrees(2 * np.arctan(np.exp(yy / EARTH_RADIUS)) - np.pi / 2)

    result = griddata(pts, vals, (lon_grid, lat_grid), method=method)
    return result


# Gleiches Ziel-Pixelraster wie in warp_o320_to_webmercator (muss mit
# WEBMERCATOR_WIDTH uebereinstimmen, damit die Indizes exakt passen) -
# einmalig ausserhalb der Schleife berechnet, da pro Lauf identisch.
_full_x_new, _full_y_new = webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH)

_gbx_min, _gby_min = lonlat_to_webmercator(GERMANY_BBOX_LONLAT[0], GERMANY_BBOX_LONLAT[2])
_gbx_max, _gby_max = lonlat_to_webmercator(GERMANY_BBOX_LONLAT[1], GERMANY_BBOX_LONLAT[3])

# Indizes im vollen Raster, die die Bbox gerade so umschliessen (lieber
# ein Pixel zu viel als zu wenig - daher aussen aufrunden statt clippen).
_col_i0 = max(0, np.searchsorted(_full_x_new, _gbx_min, side="left") - 1)
_col_i1 = min(len(_full_x_new) - 1, np.searchsorted(_full_x_new, _gbx_max, side="right"))
_row_i0 = max(0, np.searchsorted(_full_y_new, _gby_min, side="left") - 1)
_row_i1 = min(len(_full_y_new) - 1, np.searchsorted(_full_y_new, _gby_max, side="right"))

# Exakte Mercator-Extent des zugeschnittenen Rasters (= tatsaechliche
# Gitterpunkte an den Raendern, nicht die rohe Bbox - damit die
# Ruecktransformation im Frontend pixelgenau bleibt).
GERMANY_CROP_EXTENT_3857 = [
    float(_full_x_new[_col_i0]), float(_full_y_new[_row_i0]),
    float(_full_x_new[_col_i1]), float(_full_y_new[_row_i1]),
]


def crop_to_germany(data_south_first):
    """data_south_first: 2D-Array wie von warp_o320_to_webmercator
    zurückgegeben (row0 = Süden, aufsteigend in Mercator-Y wie
    _full_y_new). Schneidet auf die Deutschland-Bbox zu."""
    return data_south_first[_row_i0:_row_i1 + 1, _col_i0:_col_i1 + 1]


def data_to_rgba(data, cmap, norm):
    rgba = cmap(norm(data))
    rgba = (rgba * 255).astype(np.uint8)
    rgba[~np.isfinite(data), 3] = 0
    return rgba


def save_transparent_webp(data, cmap, norm, out_path):
    rgba = data_to_rgba(data, cmap, norm)
    img = Image.fromarray(rgba[::-1, :, :], mode="RGBA")  # Zeile 0 -> oben = Norden
    img.save(out_path, format="WEBP", lossless=True, method=4)


def embed_data_chunk(webp_path, data, extent_3857, quantum, fourcc=DVAL_FOURCC):
    """Hängt ein rohes Datenfeld als privaten, int16-quantisierten RIFF-Chunk
    an ein WebP an. Kann mehrfach mit unterschiedlichem `fourcc` aufgerufen
    werden, um mehrere Datenfelder (z.B. Mean + Spread) in dieselbe Datei
    einzubetten.

    data: 2D-Array (float), row0 = Norden (also bereits wie fürs Bild
          gespiegelt).
    extent_3857: [x_min, y_min, x_max, y_max] in Web-Mercator-Metern -
                 exakt das Raster, auf dem `data` liegt.
    quantum: Rasterschritt in den Originaleinheiten (z.B. 0.05 für °C).
    """
    height, width = data.shape

    nan_mask = ~np.isfinite(data)
    data_filled = np.where(nan_mask, 0.0, data)  # verhindert NaN->int Warnung beim Runden/Casten
    quant = np.round(data_filled / quantum)
    # Sicherheitsclip: verhindert einen int16-Überlauf bei extremen
    # Ausreißern, ohne das eigentlich zulässige Wertespektrum
    # (t2m/wind liegen weit darunter) einzuschränken.
    quant = np.clip(quant, -32767, 32767).astype(np.int16)
    quant[nan_mask] = NAN_SENTINEL_I16

    header = struct.pack("<BBII", 2, 1, width, height)
    header += struct.pack("<4d", *extent_3857)
    header += struct.pack("<d", quantum)
    compressed = zlib.compress(np.ascontiguousarray(quant, dtype="<i2").tobytes(), level=9)
    payload = header + compressed

    size = len(payload)
    chunk = fourcc + struct.pack("<I", size) + payload
    if size % 2 == 1:
        chunk += b"\x00"  # RIFF-Padding auf gerade Länge, zählt nicht zu size

    with open(webp_path, "rb") as f:
        content = f.read()

    if content[0:4] != b"RIFF" or content[8:12] != b"WEBP":
        raise ValueError(f"{webp_path} ist keine gültige WebP-Datei (RIFF/WEBP-Header fehlt)")

    riff_size = struct.unpack("<I", content[4:8])[0]
    new_riff_size = riff_size + len(chunk)

    with open(webp_path, "wb") as f:
        f.write(content[:4])
        f.write(struct.pack("<I", new_riff_size))
        f.write(content[8:])
        f.write(chunk)


def load_valid_times(root, ntime, om_path):
    """Liest die echten (nicht-gleichmaessigen) Zeitschritte direkt aus dem
    'time'-Kind der om-Datei (Unix-Timestamps in Sekunden, UTC)."""
    try:
        time_child = root.get_child_by_name("time")
    except Exception as e:
        raise ValueError(
            f"{om_path}: kein 'time'-Kind gefunden - kann Zeitstempel nicht bestimmen ({e})"
        )

    if not time_child.is_array or time_child.shape[0] != ntime:
        raise ValueError(
            f"{om_path}: 'time'-Kind passt nicht (shape={time_child.shape}, erwartet ntime={ntime})"
        )

    raw = time_child.read_array((slice(0, ntime),))
    return [dt.datetime.fromtimestamp(int(t), tz=dt.timezone.utc) for t in raw]


def load_spread_dataset(spread_var_type, spread_data_dir, expected_npoints):
    """Sucht die zur Spread-Variable passende .om-Datei in spread_data_dir,
    laedt sie komplett und gibt (data_all_spread[npoints, ntime],
    {timestamp: col_index}) zurueck.

    Gibt (None, {}) zurueck (statt zu werfen), wenn die Datei fehlt oder
    nicht zum Gitter passt - der Aufrufer erzeugt dann einfach kein
    Spread-Chunk fuer diesen Lauf, ohne den Mean-Export abzubrechen.
    """
    pattern = OM_FILENAME_PATTERNS[spread_var_type]
    try:
        spread_files = sorted(f for f in os.listdir(spread_data_dir) if f.endswith(".om"))
    except OSError as e:
        print(f"  Warnung: Spread-Ordner '{spread_data_dir}' nicht lesbar ({e}) - "
              f"kein Spread-Chunk wird eingebettet")
        return None, {}

    matches = [f for f in spread_files if pattern.lower() in f.lower()]
    if not matches:
        print(f"  Warnung: keine {spread_var_type}-Datei in '{spread_data_dir}' gefunden "
              f"(Muster '{pattern}') - kein Spread-Chunk wird eingebettet")
        return None, {}

    spread_path = os.path.join(spread_data_dir, matches[0])
    with OmFileReader(spread_path) as sroot:
        if not sroot.is_array:
            print(f"  Warnung: {spread_path} ist kein Array - kein Spread-Chunk wird eingebettet")
            return None, {}

        _, s_npoints, s_ntime = sroot.shape
        if s_npoints != expected_npoints:
            print(f"  Warnung: {spread_path} hat {s_npoints} Punkte, erwartet {expected_npoints} "
                  f"(passt nicht zum Mean-Gitter) - kein Spread-Chunk wird eingebettet")
            return None, {}

        data_all_spread = sroot.read_array((slice(0, 1), slice(0, s_npoints), slice(0, s_ntime)))[0]
        valid_times_spread = load_valid_times(sroot, s_ntime, spread_path)

    time_to_idx = {t: i for i, t in enumerate(valid_times_spread)}
    return data_all_spread, time_to_idx


# ------------------------------
# Farb-/Konvertierungs-Auswahl fuer den angeforderten var_type
# ------------------------------
if var_type not in COLORMAPS:
    print(f"Unbekannter var_type {var_type}")
    sys.exit(1)

cmap, norm = COLORMAPS[var_type]
convert = UNIT_CONVERT.get(var_type, lambda v: v)

pattern = OM_FILENAME_PATTERNS.get(var_type)
if pattern is None:
    print(f"var_type '{var_type}' hat noch kein Dateinamen-Muster in OM_FILENAME_PATTERNS")
    sys.exit(1)

# Koordinaten des O320-Gitters einmalig berechnen (unabhaengig von der
# jeweiligen Datei - gleiche Gitterdefinition fuer alle Zeitschritte/Dateien).
_o320_lat, _o320_lon = build_octahedral_latlon(GAUSSIAN_N)

# ------------------------------
# Dateien durchgehen
# ------------------------------
all_files_global = sorted(f for f in os.listdir(data_dir) if f.endswith(".om"))
matching_files = [f for f in all_files_global if pattern.lower() in f.lower()]

if not matching_files:
    print(f"Keine .om Datei in {data_dir} gefunden, die zu '{pattern}' passt "
          f"(gefunden: {all_files_global})")

spread_var_type = MEAN_TO_SPREAD_VAR.get(var_type)

for filename in matching_files:
    om_path = os.path.join(data_dir, filename)

    with OmFileReader(om_path) as root:
        if not root.is_array:
            print(f"{om_path}: root ist kein Array (is_group={root.is_group}) - überspringe")
            continue

        nmember, npoints, ntime = root.shape  # (1, 421120, ntime) fuer O320

        if npoints != _o320_lat.shape[0]:
            print(
                f"{om_path}: Anzahl Gitterpunkte ({npoints}) passt nicht zum "
                f"O{GAUSSIAN_N}-Gitter ({_o320_lat.shape[0]}) - überspringe"
            )
            continue

        # Ganzen Datensatz (alle Punkte, alle Zeitschritte) fuer den einen
        # Member lesen. Reduced-Gaussian-Punkte liegen nicht kontinuierlich
        # in der Bbox, daher zuschneiden erst danach in numpy.
        data_all = root.read_array((
            slice(0, 1),
            slice(0, npoints),
            slice(0, ntime),
        ))[0]  # shape: (npoints, ntime)

        valid_times_utc = load_valid_times(root, ntime, om_path)

        # Passende Spread-Datei (falls fuer diesen var_type vorgesehen)
        # einmal pro Mean-Datei laden - nicht pro Zeitschritt neu oeffnen.
        spread_data_all = None
        spread_time_to_idx = {}
        if spread_var_type is not None:
            spread_data_all, spread_time_to_idx = load_spread_dataset(
                spread_var_type, spread_data_dir, npoints
            )

        if var_type == "t2m_spread":
            _sample_max = np.nanmax(data_all)
            # Hinweis: t2m_spread ist eine Differenz/Streuung, keine absolute
            # Temperatur. Bei einer Differenz sind K und °C zahlengleich
            # (Versatz von 273.15 hebt sich weg) - dieser Print dient nur der
            # Kontrolle, ob der Rohwert ueberhaupt im erwarteten Bereich der
            # eigenen Bounds (0.1-18) liegt, NICHT ob konvertiert werden muss.
            print(f"{filename}: t2m_spread Rohwert-Max={_sample_max:.2f} "
                  f"(erwarteter Bereich fuer die Bounds: ~0.1-18, unabhaengig von K/°C)")

        for t_idx in range(ntime):
            data_flat = convert(np.asarray(data_all[:, t_idx], dtype=np.float64))

            # ------------------------------
            # Nach EPSG:3857 (Web Mercator) umprojizieren
            # ------------------------------
            render_data_merc = warp_o320_to_webmercator(
                data_flat, _o320_lat, _o320_lon, extent, method="linear"
            )

            # ------------------------------
            # Transparentes WebP speichern
            # ------------------------------
            outname = f"{var_type}_{valid_times_utc[t_idx].astimezone(ZoneInfo('Europe/Berlin')):%Y%m%d_%H%M}.webp"
            out_path = os.path.join(output_dir, outname)
            save_transparent_webp(render_data_merc, cmap, norm, out_path)

            # Fuer t2m/wind zusaetzlich die echten physikalischen Werte
            # (°C bzw. km/h, nicht die Farben) als privaten RIFF-Chunk
            # direkt ins WebP einbetten - row0 = Norden, damit der Chunk
            # 1:1 zur Bildorientierung passt (das Bild wird in
            # save_transparent_webp beim Speichern gespiegelt,
            # render_data_merc selbst hat row0 = Sueden).
            if var_type in EMBED_DATA_VARS:
                germany_data = crop_to_germany(render_data_merc)          # row0 = Süden
                quantum = QUANTUM_STEP.get(var_type, 0.1)
                embed_data_chunk(out_path, germany_data[::-1], GERMANY_CROP_EXTENT_3857, quantum)  # row0 = Norden

            # Zusaetzlich: passenden Spread-Wert (gleicher Zeitstempel) als
            # zweiten Chunk (DSPR) in dieselbe Datei einbetten - kein
            # eigenes Spread-Bild mehr noetig.
            if spread_data_all is not None:
                ts = valid_times_utc[t_idx]
                s_idx = spread_time_to_idx.get(ts)
                if s_idx is None:
                    print(f"  Warnung: kein {spread_var_type}-Zeitschritt fuer {ts.isoformat()} "
                          f"gefunden - kein Spread-Chunk fuer {outname} eingebettet")
                else:
                    spread_convert = UNIT_CONVERT.get(spread_var_type, lambda v: v)
                    spread_flat = spread_convert(np.asarray(spread_data_all[:, s_idx], dtype=np.float64))
                    spread_merc = warp_o320_to_webmercator(
                        spread_flat, _o320_lat, _o320_lon, extent, method="linear"
                    )
                    spread_germany = crop_to_germany(spread_merc)          # row0 = Süden
                    spread_quantum = QUANTUM_STEP.get(spread_var_type, 0.1)
                    embed_data_chunk(
                        out_path, spread_germany[::-1], GERMANY_CROP_EXTENT_3857,
                        spread_quantum, fourcc=SPREAD_CHUNK_FOURCC
                    )

            print(f"{filename} t_idx={t_idx} -> {outname}")
