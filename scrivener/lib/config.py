"""Style loading, configuration merge, font manifest, and proportional spacing."""

import json
import re
import sys
import urllib.request
from pathlib import Path

import yaml
from reportlab.lib.pagesizes import (
    A3, A4, A5, A6,
    B4, B5,
    HALF_LETTER, JUNIOR_LEGAL, LEDGER, LEGAL, LETTER, TABLOID,
    GOV_LEGAL, GOV_LETTER,
    legal, letter,
)
from reportlab.lib.units import mm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLES_DIR = PROJECT_ROOT / "styles"
IMAGES_DIR = PROJECT_ROOT / "images"
FONTS_DIR = PROJECT_ROOT / ".fonts"
MANIFEST_PATH = PROJECT_ROOT / "fonts.yaml"

_in = 72
_mm20 = 20 * mm
_mm15 = 15 * mm
_mm10 = 10 * mm

PAGE_CONFIGS = {
    "letter":       {"size": letter,        "margin": _in},
    "legal":        {"size": legal,         "margin": _in},
    "half-letter":  {"size": HALF_LETTER,   "margin": 54},
    "junior-legal": {"size": JUNIOR_LEGAL,  "margin": 54},
    "tabloid":      {"size": TABLOID,       "margin": _in},
    "ledger":       {"size": LEDGER,        "margin": _in},
    "gov-letter":   {"size": GOV_LETTER,    "margin": _in},
    "gov-legal":    {"size": GOV_LEGAL,     "margin": _in},
    "a3":           {"size": A3,            "margin": _mm20},
    "a4":           {"size": A4,            "margin": _mm20},
    "a5":           {"size": A5,            "margin": _mm15},
    "a6":           {"size": A6,            "margin": _mm10},
    "b4":           {"size": B4,            "margin": _mm20},
    "b5":           {"size": B5,            "margin": _mm15},
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".gif"}


def sp(cfg, r):
    """Spacing proportional to body_size."""
    return cfg.get("body_size", 11) * r


def deep_merge(base, override):
    """Recursively merge override into base. Dicts merge; scalars/lists replace."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_style_path(style_arg):
    """Resolve a --style argument to a .yaml file path."""
    if style_arg is None:
        return STYLES_DIR / "default.yaml"
    p = Path(style_arg)
    if p.is_file():
        return p
    candidate = STYLES_DIR / f"{style_arg}.yaml"
    if candidate.is_file():
        return candidate
    candidate = STYLES_DIR / style_arg
    if candidate.is_file():
        return candidate
    print(f"error: style not found: {style_arg}", file=sys.stderr)
    sys.exit(1)


def load_style(style_path, _loading=None):
    """Load a style YAML file with inheritance support."""
    style_path = Path(style_path)

    if _loading is None:
        _loading = set()
    key = str(style_path.resolve())
    if key in _loading:
        print(f"error: circular style inheritance involving {style_path.stem}",
              file=sys.stderr)
        sys.exit(1)
    _loading.add(key)

    with open(style_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_name = raw.pop("inherits", None)
    if base_name:
        base_path = STYLES_DIR / f"{base_name}.yaml"
        if not base_path.exists():
            print(f"error: base style '{base_name}' not found", file=sys.stderr)
            sys.exit(1)
        base = load_style(base_path, _loading)
        style = deep_merge(base, raw)
    else:
        style = raw

    resolve_palette(style)
    return style


def load_font_manifest():
    """Load fonts.yaml keyed by logical id."""
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH) as f:
        entries = yaml.safe_load(f) or []
    return {e["id"]: {"file": e["file"], "url": e.get("url")}
            for e in entries if "id" in e and "file" in e}


def resolve_font_files(style, manifest):
    """Resolve font: logical ids to file: filenames in the style's fonts config."""
    fonts_cfg = style.get("fonts", {})
    for entry in fonts_cfg.values():
        if not isinstance(entry, dict):
            continue
        font_id = entry.pop("font", None)
        if font_id and "file" not in entry:
            info = manifest.get(font_id)
            if not info:
                print(f"error: unknown font id '{font_id}' "
                      f"(not in fonts.yaml)", file=sys.stderr)
                sys.exit(1)
            entry["file"] = info["file"]


def apply_options(style, options_dict):
    """Apply user option overrides to the style config."""
    schema = style.get("options", [])
    valid_ids = {opt["id"] for opt in schema}
    for key, value in options_dict.items():
        if key not in valid_ids:
            valid = ", ".join(sorted(valid_ids)) if valid_ids else "(none)"
            print(f"error: unknown option '{key}'. valid options: {valid}",
                  file=sys.stderr)
            sys.exit(1)
        style[key] = value


def ensure_fonts_downloaded(style, manifest):
    """Download missing fonts into the shared .fonts/ cache."""
    fonts_cfg = style.get("fonts", {})
    needed = {}
    for entry in fonts_cfg.values():
        if not isinstance(entry, dict):
            continue
        fname = entry.get("file")
        if fname and fname not in needed:
            font_id = None
            for mid, minfo in manifest.items():
                if minfo["file"] == fname:
                    font_id = mid
                    break
            url = (manifest.get(font_id, {}).get("url")
                   if font_id else None) or entry.get("url")
            needed[fname] = url
    if not needed:
        return FONTS_DIR
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in needed.items():
        dest = FONTS_DIR / fname
        if dest.exists():
            continue
        if not url:
            print(f"error: font '{fname}' not found and no URL available",
                  file=sys.stderr)
            sys.exit(1)
        print(f"  fetch: {fname} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, str(dest))
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"ok ({size_mb:.1f}M)")
        except Exception as e:
            dest.unlink(missing_ok=True)
            print(f"FAILED: {e}", file=sys.stderr)
            sys.exit(1)
    return FONTS_DIR


def list_images():
    """List image files in the shared images/ directory."""
    if not IMAGES_DIR.is_dir():
        return []
    return sorted(f.name for f in IMAGES_DIR.iterdir()
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def list_styles():
    """Scan styles/ for .yaml files and return JSON-serializable catalog."""
    images = list_images()
    result = []
    if not STYLES_DIR.is_dir():
        return result
    for f in sorted(STYLES_DIR.iterdir()):
        if not f.is_file() or f.suffix != ".yaml":
            continue
        with open(f) as fh:
            raw = yaml.safe_load(fh) or {}

        base_name = raw.get("inherits")
        if base_name:
            merged = load_style(f)
        else:
            merged = raw

        style_id = f.stem
        entry = {
            "id": style_id,
            "name": merged.get("name", style_id),
            "description": merged.get("description", ""),
        }
        if base_name:
            entry["inherits"] = base_name

        options = []
        for opt in merged.get("options", []):
            o = {
                "id": opt.get("id"),
                "label": opt.get("label", opt.get("id", "")),
                "type": opt.get("type", "string"),
                "default": opt.get("default"),
            }
            if opt.get("choices"):
                o["choices"] = opt["choices"]
            options.append(o)
        if options:
            entry["options"] = options
        entry["images"] = images
        result.append(entry)
    return result


def resolve_palette(style):
    """Resolve @name color references using style.palette."""
    palette = style.get("palette", {})
    if not palette:
        return
    def resolve(obj):
        if isinstance(obj, str) and obj.startswith("@"):
            key = obj[1:]
            if key in palette:
                return palette[key]
        return obj
    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(resolve(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(resolve(v)) for v in obj]
        return obj
    resolved = walk(style)
    style.clear()
    style.update(resolved)


def extract_front_matter(md_text):
    """Split YAML front matter from body text."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", md_text, re.DOTALL)
    if not m:
        return {}, md_text
    fm = yaml.safe_load(m.group(1)) or {}
    body = md_text[m.end():]
    return fm, body


_FM_STYLE_KEYS = {"logo", "toc", "accent_saturated", "accent_mid"}


def merge_config(cli_cfg, front_matter, style):
    """Merge CLI overrides, front matter, and style into a single config dict."""
    cfg = dict(style)
    for k, v in front_matter.items():
        if k in _FM_STYLE_KEYS and v is not None:
            cfg[k] = v
    if cli_cfg.get("logo"):
        cfg["logo"] = cli_cfg["logo"]
    if cli_cfg.get("toc") is True:
        cfg["toc"] = True
    if cli_cfg.get("no_toc") is True:
        cfg["toc"] = False
    return cfg


def load_logo(path, height):
    """Load a logo image (SVG or raster) and return a ReportLab flowable."""
    from reportlab.platypus import Image as RLImage
    path = str(path)
    if path.lower().endswith(".svg"):
        from svglib.svglib import svg2rlg
        drawing = svg2rlg(path)
        if drawing is None:
            return None
        scale = height / drawing.height
        drawing.width *= scale
        drawing.height = height
        drawing.scale(scale, scale)
        return drawing
    return RLImage(path, height=height, kind="proportional")
