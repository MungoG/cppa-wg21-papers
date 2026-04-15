# tomd

Convert PDF and HTML papers to Markdown. Used to prepare WG21 inputs for
the C++ Alliance paper pipeline.

## Install

From this directory:

```
pip install -e .
```

Installs the `tomd` console script and pins
`pymupdf~=1.27` / `beautifulsoup4~=4.14`.

## Usage

```
tomd paper.pdf                  # -> paper.md (+ paper.prompts.md if uncertain)
tomd paper.html                 # -> paper.md
tomd *.pdf *.html --outdir out/ # batch mode
```

Also runnable as `python -m tomd.main ...`.

## Development

Install test extras and run the suite:

```
pip install -e .[test]
pytest tests/
```
