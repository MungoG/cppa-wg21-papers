"""Syntax highlighting via Pygments, outputting ReportLab XML markup."""

from . import escape_xml

try:
    from pygments import highlight as _hl
    from pygments.formatter import Formatter
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.token import (
        Comment, Error, Generic, Keyword, Literal, Name,
        Number, Operator, Punctuation, String, Token,
    )
    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False

_TOKEN_MAP = {
    Keyword:            "keyword",
    Keyword.Type:       "type",
    Name.Builtin:       "type",
    Name.Class:         "type",
    Name.Function:      "function",
    Name.Decorator:     "function",
    Name.Namespace:     "type",
    String:             "string",
    Literal.String:     "string",
    Number:             "number",
    Literal.Number:     "number",
    Comment:            "comment",
    Comment.Preproc:    "preprocessor",
    Operator:           "operator",
    Punctuation:        "operator",
}


def _resolve(tok, colors):
    """Walk the token type hierarchy to find a matching color."""
    t = tok
    while t:
        key = _TOKEN_MAP.get(t)
        if key and key in colors:
            return colors[key]
        t = t.parent
    return None


class _RLFormatter(Formatter):
    """Pygments formatter that emits ReportLab paragraph XML."""
    def __init__(self, colors, **kw):
        super().__init__(**kw)
        self.colors = colors

    def format(self, tokensource, outfile):
        for ttype, value in tokensource:
            text = escape_xml(value)
            color = _resolve(ttype, self.colors)
            if color:
                outfile.write(f'<font color="{color}">{text}</font>')
            else:
                outfile.write(text)


def highlight(code, lang, colors):
    """Highlight code, returning ReportLab XML markup.

    Falls back to plain escaped text if Pygments is unavailable
    or the language is not recognized.
    """
    if not HAS_PYGMENTS or not colors:
        return escape_xml(code)

    try:
        if lang:
            lexer = get_lexer_by_name(lang, stripall=True)
        else:
            lexer = guess_lexer(code)
    except Exception:
        return escape_xml(code)

    fmt = _RLFormatter(colors)
    return _hl(code, lexer, fmt)
