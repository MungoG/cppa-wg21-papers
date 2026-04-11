"""Font cache, variable-font instantiation, and ReportLab registration."""

from io import BytesIO

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

_cache = {}
_cmap_cache = {}
_lazy = {}
_families = set()
_fonts_dir = None


def set_fonts_dir(path):
    global _fonts_dir
    _fonts_dir = path


def _resolve(file_name):
    return _fonts_dir / file_name


def ensure_font(name, var_path, axes):
    if name in _cache:
        return
    if not axes:
        pdfmetrics.registerFont(TTFont(name, str(var_path)))
        _cache[name] = True
        return
    from fontTools.ttLib import TTFont as FTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    vf = FTFont(str(var_path))
    instantiateVariableFont(vf, axes, inplace=True)
    ps_name = name.replace(" ", "-")
    for rec in vf["name"].names:
        if rec.nameID == 6:
            rec.string = ps_name
    buf = BytesIO()
    vf.save(buf)
    buf.seek(0)
    try:
        pdfmetrics.registerFont(TTFont(name, buf))
    except Exception:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".ttf", delete=False)
        buf.seek(0)
        tmp.write(buf.read())
        tmp.flush()
        pdfmetrics.registerFont(TTFont(name, tmp.name))
    _cache[name] = True


def get_cmap(var_path, axes):
    key = (str(var_path), tuple(sorted(axes.items())))
    if key in _cmap_cache:
        return _cmap_cache[key]
    from fontTools.ttLib import TTFont as FTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    vf = FTFont(str(var_path))
    instantiateVariableFont(vf, axes, inplace=True)
    cmap = set()
    for table in vf["cmap"].tables:
        if table.cmap:
            cmap.update(table.cmap.keys())
    _cmap_cache[key] = cmap
    return cmap


def ensure_lazy(name):
    if name in _cache:
        return
    if name not in _lazy:
        return
    var_path, axes = _lazy[name]
    ensure_font(name, var_path, axes)


def register_fonts(style):
    fonts_cfg = style.get("fonts", {})
    font_map = {
        "body": "Body",
        "body_bold": "Body-Bold",
        "body_italic": "Body-Italic",
        "body_bold_italic": "Body-BoldItalic",
        "code": "Code",
        "code_bold": "Code-Bold",
        "code_italic": "Code-Italic",
        "code_bold_italic": "Code-BoldItalic",
        "cjk": "CJK",
    }
    eager = {"body", "body_bold"}
    for key, rl_name in font_map.items():
        entry = fonts_cfg.get(key)
        if not entry:
            continue
        path = _resolve(entry["file"])
        axes = dict(entry.get("axes", {}))
        if key in eager:
            ensure_font(rl_name, path, axes)
        else:
            _lazy[rl_name] = (path, axes)


def register_families():
    if "Body" not in _families:
        ensure_lazy("Body-Italic")
        ensure_lazy("Body-BoldItalic")
        pdfmetrics.registerFontFamily(
            "Body", normal="Body", bold="Body-Bold",
            italic="Body-Italic", boldItalic="Body-BoldItalic")
        _families.add("Body")


def ensure_code_family():
    if "Code" in _families:
        return
    ensure_lazy("Code")
    ensure_lazy("Code-Bold")
    ensure_lazy("Code-Italic")
    ensure_lazy("Code-BoldItalic")
    pdfmetrics.registerFontFamily(
        "Code", normal="Code", bold="Code-Bold",
        italic="Code-Italic", boldItalic="Code-BoldItalic")
    _families.add("Code")


def build_body_cmap(style):
    entry = style.get("fonts", {}).get("body", {})
    if not entry:
        return set()
    path = _resolve(entry["file"])
    axes = dict(entry.get("axes", {}))
    return get_cmap(path, axes)
