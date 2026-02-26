#!/usr/bin/env bash
# render.sh - Render WG21 papers from markdown to HTML and PDF.
#
# Usage: render.sh <master-checkout> <rendered-worktree>
#
# Processes source/d*.md and archive/p*.md from the master checkout,
# copies markdown into the rendered worktree, and generates HTML + PDF
# in the rendered worktree root.  Skips files whose content has not
# changed since the last rendered commit.
#
# Can be run locally (Git Bash, macOS, Linux) or from CI.

set -uo pipefail

MASTER="${1:?Usage: render.sh <master-checkout> <rendered-worktree>}"
RENDERED="${2:?Usage: render.sh <master-checkout> <rendered-worktree>}"

# Resolve to absolute paths
MASTER="$(cd "$MASTER" && pwd)"
RENDERED="$(cd "$RENDERED" && pwd)"
TOOLS="$MASTER/tools"

# ── Detect platform ───────────────────────────────────────────

detect_chrome() {
    if [ "${OS:-}" = "Windows_NT" ]; then
        for candidate in \
            "${LOCALAPPDATA:-}/Google/Chrome/Application/chrome.exe" \
            "C:/Program Files/Google/Chrome/Application/chrome.exe" \
            "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"; do
            if [ -f "$candidate" ]; then
                echo "$candidate"
                return
            fi
        done
    elif [ "$(uname -s)" = "Darwin" ]; then
        for candidate in \
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
            "/Applications/Chromium.app/Contents/MacOS/Chromium"; do
            if [ -x "$candidate" ]; then
                echo "$candidate"
                return
            fi
        done
    else
        for candidate in google-chrome chromium-browser chromium; do
            if command -v "$candidate" > /dev/null 2>&1; then
                echo "$candidate"
                return
            fi
        done
    fi
    return 1
}

if [ "${OS:-}" = "Windows_NT" ]; then
    MERMAID_FILTER="mermaid-filter.cmd"
else
    MERMAID_FILTER="mermaid-filter"
fi

CHROME="$(detect_chrome)" || {
    echo "ERROR: Google Chrome or Chromium not found"
    exit 1
}

echo "Platform:       $(uname -s)"
echo "Chrome:         $CHROME"
echo "Mermaid filter: $MERMAID_FILTER"
echo "Master:         $MASTER"
echo "Rendered:       $RENDERED"
echo ""

# ── Track failures ────────────────────────────────────────────

FAILURES=0

# ── Helper: extract YAML document field ───────────────────────

extract_document_field() {
    sed -n '/^---$/,/^---$/p' "$1" \
        | sed -n 's/^document:[[:space:]]*//p' \
        | tr -d '[:space:]'
}

# ── Helper: render markdown to HTML ───────────────────────────

render_html() {
    local mdfile="$1"    # relative to $RENDERED, e.g. source/d2583.md
    local htmlfile="$2"  # relative to $RENDERED, e.g. d2583.html

    cp "$TOOLS/mermaid-config.json" "$RENDERED/.mermaid-config.json"
    (
        cd "$RENDERED"
        MERMAID_FILTER_FORMAT=svg pandoc --standalone \
            --filter "$MERMAID_FILTER" \
            --embed-resources --toc \
            --template="$TOOLS/wg21.html5" \
            --css="$TOOLS/paperstyle.css" \
            -o "$htmlfile" "$mdfile"
    )
    local rc=$?
    rm -f "$RENDERED/.mermaid-config.json"
    [ ! -s "$RENDERED/mermaid-filter.err" ] \
        && rm -f "$RENDERED/mermaid-filter.err" 2>/dev/null || true
    return $rc
}

# ── Helper: render HTML to PDF ────────────────────────────────

render_pdf() {
    local htmlfile="$1"  # absolute path
    local pdffile="$2"   # absolute path

    # On MSYS/Cygwin, convert to a Windows path for the file:// URL.
    # MSYS auto-converts bare path arguments but not paths embedded
    # inside a URL string.
    local html_url
    if command -v cygpath > /dev/null 2>&1; then
        html_url="file://$(cygpath -m "$htmlfile")"
    else
        html_url="file://$htmlfile"
    fi

    "$CHROME" --headless --no-pdf-header-footer \
        --run-all-compositor-stages-before-draw \
        --disable-gpu --no-sandbox \
        --print-to-pdf="$pdffile" \
        "$html_url" 2> >(grep -v 'dbus/' >&2)
}

# ── Process a directory ───────────────────────────────────────
# Args: $1 = subdir name ("source" or "archive")
#        $2 = glob pattern ("d*.md" or "p*.md")
#        $3 = "validate" or "skip" (YAML check)

process_dir() {
    local subdir="$1"
    local glob="$2"
    local validate="$3"

    local src_dir="$MASTER/$subdir"
    local dst_dir="$RENDERED/$subdir"

    mkdir -p "$dst_dir"

    # Collect matching markdown files
    local md_files=()
    for f in "$src_dir"/$glob; do
        [ -f "$f" ] || continue
        md_files+=("$(basename "$f")")
    done

    if [ ${#md_files[@]} -eq 0 ]; then
        echo "No $glob files in $subdir/"
        return
    fi

    echo "=== Processing $subdir/ (${#md_files[@]} file(s)) ==="

    for mdname in "${md_files[@]}"; do
        local stem="${mdname%.md}"
        local md_dst="$dst_dir/$mdname"
        local html_dst="$RENDERED/$stem.html"
        local pdf_dst="$RENDERED/$stem.pdf"

        echo ""
        echo "-- $subdir/$mdname --"

        # Copy markdown to rendered worktree
        cp "$src_dir/$mdname" "$md_dst"

        # ── YAML validation (archive only) ────────────────
        if [ "$validate" = "validate" ]; then
            local doc_field
            doc_field="$(extract_document_field "$md_dst")" || true
            local doc_lower
            doc_lower="$(echo "$doc_field" | tr '[:upper:]' '[:lower:]')"
            local file_prefix="${stem%%-*}"

            if [ -z "$doc_field" ]; then
                echo " "
                echo " "
                echo -e "  \033[31m*** ERROR\033[0m"
                echo -e "  \033[31m*** ERROR: no document field in YAML front matter\033[0m"
                echo -e "  \033[31m*** ERROR\033[0m"
                echo " "
                echo " "
                FAILURES=$((FAILURES + 1))
                continue
            fi

            if [ "$doc_lower" != "$file_prefix" ]; then
                echo " "
                echo " "
                echo -e "  \033[31m*** ERROR\033[0m"
                echo -e "  \033[31m*** ERROR: document '$doc_field' does not match filename prefix '$file_prefix'\033[0m"
                echo -e "  \033[31m*** ERROR\033[0m"
                echo " "
                echo " "
                FAILURES=$((FAILURES + 1))
                continue
            fi
            echo "  YAML OK: $doc_field"
        fi

        # ── Detect if HTML needs regeneration ─────────────
        local need_html=false
        local need_pdf=false

        # Check if the markdown content changed in the rendered worktree
        if (cd "$RENDERED" && git diff --quiet -- "$subdir/$mdname") 2>/dev/null; then
            # File matches what is committed - but is it tracked at all?
            if (cd "$RENDERED" && git ls-files --error-unmatch "$subdir/$mdname") > /dev/null 2>&1; then
                echo "  Markdown unchanged"
            else
                echo "  New file"
                need_html=true
            fi
        else
            echo "  Markdown changed"
            need_html=true
        fi

        # Force regeneration if HTML is missing
        if [ ! -f "$html_dst" ]; then
            echo "  HTML missing"
            need_html=true
        fi

        # ── Render HTML if needed ─────────────────────────
        if [ "$need_html" = true ]; then
            echo "  Generating HTML..."
            if ! render_html "$subdir/$mdname" "$stem.html"; then
                echo "  ERROR: HTML generation failed for $mdname"
                rm -f "$html_dst" "$pdf_dst"
                FAILURES=$((FAILURES + 1))
                continue
            fi
            need_pdf=true
        fi

        # Force regeneration if PDF is missing
        if [ ! -f "$pdf_dst" ]; then
            echo "  PDF missing"
            need_pdf=true
        fi

        # ── Render PDF if needed ──────────────────────────
        if [ "$need_pdf" = true ]; then
            echo "  Generating PDF..."
            if ! render_pdf "$html_dst" "$pdf_dst"; then
                echo "  ERROR: PDF generation failed for $mdname"
                rm -f "$pdf_dst"
                FAILURES=$((FAILURES + 1))
                continue
            fi
        fi

        if [ "$need_html" = false ] && [ "$need_pdf" = false ]; then
            echo "  Up to date"
        fi
    done
}

# ── Main ──────────────────────────────────────────────────────

process_dir "source"  "d*.md"  "skip"
echo ""
process_dir "archive" "p*.md"  "validate"

echo ""
if [ "$FAILURES" -gt 0 ]; then
    echo "Completed with $FAILURES failure(s)"
    exit 1
else
    echo "All papers rendered successfully"
fi
