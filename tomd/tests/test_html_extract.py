"""Tests for lib.html.extract."""

from lib.html.extract import (
    parse_html, detect_generator, extract_metadata, strip_boilerplate,
)


class TestDetectGenerator:
    def test_mpark(self):
        html = '<meta name="generator" content="mpark/wg21" />'
        assert detect_generator(parse_html(html)) == "mpark"

    def test_bikeshed(self):
        html = '<meta content="Bikeshed version abc" name="generator">'
        assert detect_generator(parse_html(html)) == "bikeshed"

    def test_hackmd_link(self):
        html = '<link href="https://hackmd.io/favicon.ico" rel="icon">'
        assert detect_generator(parse_html(html)) == "hackmd"

    def test_hackmd_title(self):
        html = "<title>HackMD doc</title>"
        assert detect_generator(parse_html(html)) == "hackmd"

    def test_hand_written(self):
        html = "<address>Author info</address>"
        assert detect_generator(parse_html(html)) == "hand-written"

    def test_mpark_fallback_header(self):
        html = '<header id="title-block-header"><h1>Title</h1></header>'
        assert detect_generator(parse_html(html)) == "mpark"

    def test_unknown(self):
        html = "<html><body><p>Hello</p></body></html>"
        assert detect_generator(parse_html(html)) == "unknown"


class TestExtractMparkMetadata:
    MPARK_HTML = """
    <header id="title-block-header">
    <h1 class="title">any_view</h1>
    <table>
      <tr><td>Document #:</td><td>P3411R5</td></tr>
      <tr><td>Date:</td><td>2026-01-25</td></tr>
      <tr><td>Audience:</td><td>SG9, LEWG</td></tr>
      <tr><td>Reply-to:</td><td>
        Alice Smith<br>&lt;<a href="mailto:alice@x.com">alice@x.com</a>&gt;
      </td></tr>
    </table>
    </header>
    """

    def test_title(self):
        soup = parse_html(self.MPARK_HTML)
        meta = extract_metadata(soup, "mpark")
        assert meta["title"] == "any_view"

    def test_document(self):
        soup = parse_html(self.MPARK_HTML)
        meta = extract_metadata(soup, "mpark")
        assert meta["document"] == "P3411R5"

    def test_date(self):
        soup = parse_html(self.MPARK_HTML)
        meta = extract_metadata(soup, "mpark")
        assert meta["date"] == "2026-01-25"

    def test_audience(self):
        soup = parse_html(self.MPARK_HTML)
        meta = extract_metadata(soup, "mpark")
        assert meta["audience"] == "SG9, LEWG"

    def test_reply_to(self):
        soup = parse_html(self.MPARK_HTML)
        meta = extract_metadata(soup, "mpark")
        assert "reply-to" in meta
        assert any("Alice" in a and "alice@x.com" in a
                    for a in meta["reply-to"])


class TestExtractBikeshedMetadata:
    BIKESHED_HTML = """
    <h1 class="p-name no-ref" id="title">P3953R0<br>Rename std::runtime_format</h1>
    <h2 class="no-num" id="profile-and-date">
      Published, <time class="dt-updated" datetime="2025-12-28">2025-12-28</time>
    </h2>
    <dl>
      <dt class="editor">Author:</dt>
      <dd class="editor"><a class="email" href="mailto:bob@x.com">Bob Jones</a></dd>
      <dt>Audience:</dt>
      <dd>LEWG</dd>
    </dl>
    """

    def test_title(self):
        soup = parse_html(self.BIKESHED_HTML)
        meta = extract_metadata(soup, "bikeshed")
        assert "Rename" in meta.get("title", "")

    def test_document(self):
        soup = parse_html(self.BIKESHED_HTML)
        meta = extract_metadata(soup, "bikeshed")
        assert meta["document"] == "P3953R0"

    def test_date(self):
        soup = parse_html(self.BIKESHED_HTML)
        meta = extract_metadata(soup, "bikeshed")
        assert meta["date"] == "2025-12-28"

    def test_audience(self):
        soup = parse_html(self.BIKESHED_HTML)
        meta = extract_metadata(soup, "bikeshed")
        assert meta.get("audience") == "LEWG"


class TestExtractHandwrittenMetadata:
    def test_address_block(self):
        html = """
        <address>
        Document number: P4005R0<br/>
        Audience: EWG<br/>
        <a href="mailto:v@g.com">Ville V</a><br/>
        2026-02-02<br/>
        </address>
        <h1>Title Here</h1>
        """
        soup = parse_html(html)
        meta = extract_metadata(soup, "hand-written")
        assert meta["document"] == "P4005R0"
        assert meta["audience"] == "EWG"
        assert meta["date"] == "2026-02-02"
        assert meta["title"] == "Title Here"


class TestStripBoilerplate:
    def test_removes_style_script(self):
        html = "<style>body{}</style><script>x()</script><p>Keep</p>"
        soup = parse_html(html)
        strip_boilerplate(soup, "mpark")
        assert soup.find("style") is None
        assert soup.find("script") is None
        assert soup.find("p") is not None

    def test_removes_toc(self):
        html = '<div id="TOC"><ul><li>Item</li></ul></div><p>Body</p>'
        soup = parse_html(html)
        strip_boilerplate(soup, "mpark")
        assert soup.find(id="TOC") is None

    def test_unknown_returns_problem(self):
        html = "<p>Hello</p>"
        soup = parse_html(html)
        problems = strip_boilerplate(soup, "unknown")
        assert len(problems) == 1
        assert "Unrecognized" in problems[0]

    def test_known_returns_no_problem(self):
        html = '<header id="title-block-header"></header><p>Body</p>'
        soup = parse_html(html)
        problems = strip_boilerplate(soup, "mpark")
        assert len(problems) == 0
