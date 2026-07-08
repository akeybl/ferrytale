#!/usr/bin/env python3
"""Build the ClubFloyd catalog and transcript cache.

Source:
  - every ClubFloyd transcript listed in the archive table on
    https://allthingsjacq.com/interactive_fiction.html#clubfloyd
    (the table that lists pages like intfic_clubfloyd_20180930.html)

ClubFloyd pages are chat logs rendered as an HTML table; only the left column
(the game session) is used — the right column is side chatter. Processing:
  - "Floyd ]" status-bar lines are removed
  - "Floyd | text" output lines keep their text (prompt echoes like "> X ME"
    are dropped; the command line itself carries the input)
  - '<player> says (to Floyd), "x me"' becomes '> x me'
  - the initial "load" command is dropped; everything Floyd prints before the
    first real player command becomes the opening page
  - any command whose game output exactly matches an earlier output is dropped
    together with that output
  - all other (non-Floyd) chat lines are dropped

The tracked catalog at catalog/clubfloyd.json maps slugs to titles, authors,
dates, and source URLs. Downloaded transcript text remains a local cache under
transcripts/.

Output format per downloaded game (transcripts/<slug>.txt): opening text, then
"> command" lines each followed by the game's response. transcripts/index.json
maps downloaded slugs to titles/authors/sources for compatibility.

Raw downloads are cached in transcripts/_raw/ (gitignored); rerunning the
script only fetches missing pages.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CATALOG_DIR = BASE_DIR / "catalog"
CATALOG_PATH = CATALOG_DIR / "clubfloyd.json"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
RAW_DIR = TRANSCRIPTS_DIR / "_raw"

ARCHIVE_URL = "https://allthingsjacq.com/interactive_fiction.html"
CLUBFLOYD_BASE = "https://allthingsjacq.com/"

FETCH_WORKERS = 3
USER_AGENT = "Ferrytale/1.0 (interactive-fiction transcript fetcher; https://discord.gg/QNDhbYWKr4)"
# Be polite to the volunteer-run archive: pause before each network fetch so a
# 652-game `--all` run trickles requests instead of bursting the server.
FETCH_DELAY_SECONDS = 1.0

# The relay bot is "Floyd" in older sessions and "CF" / "ClubFloyd" in newer
# ones; some pages lowercase the name.
FLOYD_LINE_RE = re.compile(r"^(?:Floyd|CF|ClubFloyd)\s*([|\]])\s?(.*)$", re.IGNORECASE)
SAYS_TO_FLOYD_RE = re.compile(
    r'^(?:You|[A-Za-z0-9_\-\[\]]+)\s+(?:says?|asks?|exclaims?)\s+'
    r'\(to (?:Floyd|CF|ClubFloyd)\),\s+"(.*)"\s*$',
    re.IGNORECASE,
)
LOAD_COMMAND_RE = re.compile(r"^\s*(?:load|play)\b", re.IGNORECASE)
# Page names are a date plus an optional suffix: 20180930.html, 20091203a.html
# (multi-part sessions), 20131001-NF.html (NightFloyd sessions).
ARCHIVE_LINK_RE = re.compile(
    r'<A\s+HREF="(intfic_clubfloyd_([0-9]{8}[^"]*?)\.html)"\s*>(.*?)</A>',
    re.IGNORECASE | re.DOTALL,
)
TITLE_RE = re.compile(r"<I>(.*?)</I>", re.IGNORECASE | re.DOTALL)


def html_to_text(fragment: str) -> str:
    fragment = re.sub(r"</?br\s*/?>", " ", fragment, flags=re.IGNORECASE)
    text = html.unescape(re.sub(r"<[^>]+>", "", fragment))
    return re.sub(r"[ \t\r\n\f\v]+", " ", text).strip()


def fetch(url: str, cache_name: str) -> bytes:
    cache_path = RAW_DIR / cache_name
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    # Polite inter-request delay so concurrent workers don't burst the archive.
    time.sleep(FETCH_DELAY_SECONDS)
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
            return data
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last_error}")


def decode_text(raw: bytes) -> str:
    if raw[:2] == b"\x1f\x8b":  # gzip magic — some archived snapshots arrive compressed
        raw = gzip.decompress(raw)
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("latin-1", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def slugify(title: str) -> str:
    title = re.sub(r"\[NF\]", "", title, flags=re.IGNORECASE)
    title = re.sub(r"&", " and ", title)
    title = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()
    return title or "untitled"


# ── ClubFloyd HTML: pull the left-column cell texts in document order ────────

class ClubFloydTableParser(HTMLParser):
    """Collects the text of each table row's game-session cell.

    Two layouts exist: modern pages mark cells with class="room ..." (left)
    and class="interlude" (right); old pages use positional columns
    <tr><td/><td>game</td><td/><td>chat</td><td/></tr>.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[str] = []           # finished left-column cell texts
        self._row: list[tuple[str, str]] = []  # (class, text) per td
        self._in_td = False
        self._td_class = ""
        self._td_text: list[str] = []
        self._row_has_classes = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._finish_row()
        elif tag == "td":
            self._finish_td()
            self._in_td = True
            self._td_class = dict(attrs).get("class", "") or ""
            self._td_text = []
        elif tag == "br" and self._in_td:
            self._td_text.append("\n")

    def handle_endtag(self, tag):
        if tag == "td":
            self._finish_td()
        elif tag in ("tr", "table"):
            self._finish_row()

    def handle_data(self, data):
        if self._in_td:
            self._td_text.append(data)

    def close(self):
        super().close()
        self._finish_row()

    def _finish_td(self):
        if not self._in_td:
            return
        self._row.append((self._td_class, "".join(self._td_text)))
        if self._td_class:
            self._row_has_classes = True
        self._in_td = False
        self._td_class = ""
        self._td_text = []

    def _finish_row(self):
        self._finish_td()
        if self._row:
            text = self._left_cell_text(self._row)
            if text is not None:
                self.cells.append(text)
        self._row = []
        self._row_has_classes = False

    @staticmethod
    def _left_cell_text(row: list[tuple[str, str]]) -> str | None:
        classed = [text for cls, text in row if "room" in cls.split()]
        if classed:
            return "\n".join(classed)
        if any(cls for cls, _ in row):
            # classed layout but no room cell in this row (pure interlude)
            return None
        # positional layout: [spacer, game, spacer, chat, spacer]
        if len(row) >= 2:
            return row[1][1]
        if row:
            return row[0][1]
        return None


# ── Line rules → (kind, text) items ──────────────────────────────────────────

def clubfloyd_items(page_html: str) -> list[tuple[str, str]]:
    parser = ClubFloydTableParser()
    parser.feed(page_html)
    parser.close()

    # First pass: classify lines, keeping the Floyd delimiter ("]" = status
    # window, "|" = main window).
    seq: list[tuple[str, str, str]] = []  # (kind, delim, text)
    for cell in parser.cells:
        for line in cell.split("\n"):
            line = line.rstrip()
            floyd = FLOYD_LINE_RE.match(line)
            if floyd:
                delimiter, content = floyd.groups()
                seq.append(("floyd", delimiter, content))
                continue
            says = SAYS_TO_FLOYD_RE.match(line.strip())
            if says:
                command = says.group(1).strip()
                if command:
                    seq.append(("cmd", "", command))
            # everything else is channel chatter: dropped

    # Second pass: per-turn status bars are 1-2 consecutive "]" lines — drop
    # them. Longer "]" runs are full status-window screens (banner pages,
    # Aisle's intro) — keep them. A "|" line sandwiched between "]" lines is
    # a line-wrap artifact duplicating the bracket text — drop it.
    items: list[tuple[str, str]] = []
    i = 0
    while i < len(seq):
        kind, delim, text = seq[i]
        if kind == "cmd":
            items.append(("cmd", text))
            i += 1
            continue
        if delim == "|":
            prev_is_bracket = i > 0 and seq[i - 1][0] == "floyd" and seq[i - 1][1] == "]"
            next_is_bracket = (
                i + 1 < len(seq) and seq[i + 1][0] == "floyd" and seq[i + 1][1] == "]"
            )
            if prev_is_bracket and next_is_bracket:
                i += 1  # wrap artifact inside a status-window screen
                continue
            if not text.lstrip().startswith(">"):  # drop echoed prompts/commands
                items.append(("out", text))
            i += 1
            continue
        # "]" line: measure the bracket run (skipping sandwiched "|" artifacts)
        run: list[str] = []
        j = i
        while j < len(seq) and seq[j][0] == "floyd":
            d = seq[j][1]
            if d == "]":
                run.append(seq[j][2])
                j += 1
                continue
            if d == "|" and j + 1 < len(seq) and seq[j + 1][0] == "floyd" and seq[j + 1][1] == "]":
                j += 1
                continue
            break
        if len(run) >= 3:
            for content in run:
                if not content.lstrip().startswith(">"):
                    items.append(("out", content))
        i = j
    return items


def normalize_output(lines: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def tidy_block(lines: list[str]) -> str:
    text = "\n".join(line.rstrip() for line in lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def assemble_transcript(items: list[tuple[str, str]]) -> str | None:
    """Opening (pre-first-command Floyd text), then deduped command/output pairs.

    Multi-week sessions re-`load` the game partway through; output that
    follows a dropped load command (after the opening is done) becomes a bare
    output block, deduped like any other so repeated banners disappear."""
    opening: list[str] = []
    opening_done = False
    pairs: list[tuple[str | None, list[str]]] = []
    seen_outputs: set[str] = set()
    current_cmd: str | None = None
    current_out: list[str] = []

    def flush() -> None:
        nonlocal current_cmd, current_out
        if current_cmd is None and not opening_done:
            opening.extend(current_out)
        elif current_out or current_cmd is not None:
            norm = normalize_output(current_out)
            if not norm or norm not in seen_outputs:
                if norm:
                    seen_outputs.add(norm)
                pairs.append((current_cmd, current_out))
        current_cmd = None
        current_out = []

    for kind, text in items:
        if kind == "out":
            current_out.append(text)
            continue
        if not opening_done and not opening and not current_out:
            continue  # command to the bot before the game produced anything
        flush()
        if LOAD_COMMAND_RE.match(text):
            continue  # dropped; any output it produces is handled by flush()
        current_cmd = text
        opening_done = True
    flush()

    opening_text = tidy_block(opening)
    if not opening_text and not pairs:
        return None
    blocks = [opening_text] if opening_text else []
    for command, out_lines in pairs:
        body = tidy_block(out_lines)
        if command is None:
            if body:
                blocks.append(body)
            continue
        block = f"> {command}"
        if body:
            block += "\n" + body
        blocks.append(block)
    return "\n\n".join(blocks) + "\n"


# ── Archive scrape ───────────────────────────────────────────────────────────

def archive_entries() -> list[dict]:
    raw = fetch(ARCHIVE_URL, "interactive_fiction.html")
    page = decode_text(raw)
    entries: list[dict] = []
    seen_hrefs: set[str] = set()
    for match in ARCHIVE_LINK_RE.finditer(page):
        href, date, label = match.groups()
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)
        label = re.sub(r"\s+", " ", label)
        title_match = TITLE_RE.search(label)
        title = html_to_text(title_match.group(1)) if title_match else ""
        plain = html.unescape(re.sub(r"<[^>]+>", "", label)).strip()
        if not title:
            title = plain.split(" by ")[0].strip() or href
        author_match = re.search(r"\bby\s+(.+)$", plain, re.IGNORECASE)
        author = author_match.group(1).strip() if author_match else ""
        entries.append({
            "href": href,
            "url": CLUBFLOYD_BASE + href,
            "date": date,
            "title": title,
            "author": author,
        })
    return entries


def assign_slugs(entries: list[dict], taken: set[str]) -> None:
    for entry in entries:
        slug = slugify(entry["title"])
        if slug in taken:
            slug = f"{slug}-{entry['date']}"
        suffix = 2
        base = slug
        while slug in taken:
            slug = f"{base}-{suffix}"
            suffix += 1
        taken.add(slug)
        entry["slug"] = slug


def catalog_from_entries(entries: list[dict]) -> dict[str, dict]:
    return {
        entry["slug"]: {
            "title": entry["title"],
            "author": entry["author"],
            "date": entry["date"],
            "href": entry["href"],
            "source": entry["url"],
        }
        for entry in entries
    }


def refresh_catalog() -> dict[str, dict]:
    entries = archive_entries()
    assign_slugs(entries, set())
    catalog = catalog_from_entries(entries)
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return catalog


def load_catalog(refresh: bool = False) -> dict[str, dict]:
    if refresh:
        return refresh_catalog()
    if not CATALOG_PATH.exists():
        raise RuntimeError(f"missing checked-in ClubFloyd catalog: {CATALOG_PATH}")
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def wayback_snapshot_timestamp(source: str) -> str | None:
    parsed = urllib.parse.urlparse(str(source or "").strip())
    if parsed.netloc.lower() != "web.archive.org":
        return None
    match = re.match(r"^/web/([0-9]+)[a-z_]*?/", parsed.path)
    return match.group(1) if match else None


def source_cache_name(entry: dict) -> str:
    href = str(entry.get("href") or "").strip()
    source = str(entry.get("source") or "")
    timestamp = wayback_snapshot_timestamp(source)
    if href:
        name = Path(href).name
        if timestamp:
            return f"wayback-{timestamp}-{name}"
        return name
    parsed = urllib.parse.urlparse(source)
    name = Path(parsed.path).name
    if name:
        return name
    return slugify(source) + ".html"


def transcript_metadata(entry: dict) -> dict[str, str]:
    return {
        "title": entry.get("title", ""),
        "author": entry.get("author", ""),
        "source": entry.get("source", ""),
    }


def write_local_index_entry(slug: str, entry: dict) -> None:
    path = TRANSCRIPTS_DIR / "index.json"
    index: dict[str, dict] = {}
    if path.exists():
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index = {}
    index[slug] = transcript_metadata(entry)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_local_index() -> dict[str, dict]:
    path = TRANSCRIPTS_DIR / "index.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def is_clubfloyd_source(source: str) -> bool:
    """Accept a direct ClubFloyd page URL or a Web Archive (Wayback Machine)
    snapshot of one, so the catalog can point at archived links that don't add
    load to the volunteer-run allthingsjacq.com server."""
    source = (source or "").strip()
    if source.startswith(CLUBFLOYD_BASE):
        return True
    parsed = urllib.parse.urlparse(source)
    return parsed.netloc.lower() == "web.archive.org" and CLUBFLOYD_BASE in source


def build_transcript_text(entry: dict) -> str | None:
    source = str(entry.get("source") or "")
    if not is_clubfloyd_source(source):
        raise RuntimeError(f"unsupported transcript source: {source}")
    raw = fetch(source, source_cache_name(entry))
    return assemble_transcript(clubfloyd_items(decode_text(raw)))


def page_title_metadata(page_html: str) -> tuple[str, str]:
    """Title and author from a transcript page's <title>, e.g.
    'AllThingsJacq.com - ... | ClubFloyd - January 28, 2024 - Get Corn by
    Joey Tanden' -> ('Get Corn', 'Joey Tanden')."""
    match = re.search(r"<title>(.*?)</title>", page_html, re.S | re.I)
    text = html_to_text(match.group(1)).strip() if match else ""
    part = text.split(" - ")[-1].strip() if " - " in text else text
    if " by " in part:
        title, author = part.rsplit(" by ", 1)
        return title.strip(), author.strip()
    return part, ""


def download_from_url(url: str, force: bool = False) -> tuple[str, Path]:
    """Install a transcript straight from a supported page URL — it does not
    need to be in the catalog. Supported now: ClubFloyd pages, either direct
    (allthingsjacq.com) or via a web.archive.org snapshot. Returns
    (slug, transcript_path); the slug comes from the page's own title."""
    url = str(url or "").strip()
    if not is_clubfloyd_source(url):
        raise RuntimeError(
            "unsupported URL — only ClubFloyd transcript pages are supported "
            f"for now ({CLUBFLOYD_BASE}intfic_clubfloyd_*.html, directly or "
            "through a web.archive.org snapshot)"
        )
    raw = fetch(url, source_cache_name({"source": url, "href": ""}))
    page_html = decode_text(raw)
    text = assemble_transcript(clubfloyd_items(page_html))
    if not text or len(text) < 400:
        raise RuntimeError(f"no usable transcript found at {url}")
    title, author = page_title_metadata(page_html)
    slug = slugify(title) if title else ""
    if not slug:
        slug = slugify(Path(urllib.parse.urlparse(url).path).stem)
    entry = {
        "title": title or slug.replace("-", " ").title(),
        "author": author,
        "source": url,
    }
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    path = TRANSCRIPTS_DIR / f"{slug}.txt"
    existing_source = (load_local_index().get(slug) or {}).get("source")
    if path.exists() and path.stat().st_size > 0 and not force and existing_source == url:
        return slug, path
    path.write_text(text, encoding="utf-8")
    write_local_index_entry(slug, entry)
    return slug, path


def download_game(slug: str, catalog: dict[str, dict] | None = None, force: bool = False) -> Path:
    catalog = catalog or load_catalog()
    entry = catalog.get(slug)
    if not entry:
        raise RuntimeError(f"{slug!r} is not in the ClubFloyd catalog")
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    path = TRANSCRIPTS_DIR / f"{slug}.txt"
    local_source = (load_local_index().get(slug) or {}).get("source")
    catalog_source = entry.get("source")
    if local_source and catalog_source and local_source != catalog_source:
        force = True
    if path.exists() and path.stat().st_size > 0 and not force:
        write_local_index_entry(slug, entry)
        return path
    text = build_transcript_text(entry)
    if not text or len(text) < 400:
        raise RuntimeError(f"no usable session for {slug!r}")
    path.write_text(text, encoding="utf-8")
    write_local_index_entry(slug, entry)
    return path


def build_all(catalog: dict[str, dict], force: bool = False) -> int:
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    index: dict[str, dict] = {}
    entries = [dict(info, slug=slug) for slug, info in catalog.items()]
    print(f"ClubFloyd catalog: {len(entries)} transcripts")

    def build_one(entry: dict) -> tuple[dict, str | None, str]:
        try:
            path = TRANSCRIPTS_DIR / f"{entry['slug']}.txt"
            if path.exists() and path.stat().st_size > 0 and not force:
                return entry, path.read_text(encoding="utf-8"), ""
            text = build_transcript_text(entry)
            return entry, text, ""
        except Exception as exc:
            return entry, None, str(exc)

    written = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        for entry, text, error in pool.map(build_one, entries):
            if error:
                failed += 1
                print(f"  FAIL {entry['href']}: {error}", file=sys.stderr)
                continue
            if not text or len(text) < 400:
                skipped += 1
                print(f"  skip {entry['slug']} ({entry.get('href', entry.get('source', ''))}): no usable session")
                continue
            (TRANSCRIPTS_DIR / f"{entry['slug']}.txt").write_text(text, encoding="utf-8")
            index[entry["slug"]] = transcript_metadata(entry)
            written += 1

    (TRANSCRIPTS_DIR / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    orphans = sorted(
        p.name for p in TRANSCRIPTS_DIR.glob("*.txt") if p.stem not in index
    )
    if orphans:
        print(f"stale files not in index (slug changed between builds?): {orphans}")
    print(f"written={written} skipped={skipped} failed={failed}; "
          f"index.json has {len(index)} games")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ClubFloyd transcript cache files from the checked-in catalog."
    )
    parser.add_argument("--all", action="store_true",
                        help="download every catalogued transcript into transcripts/")
    parser.add_argument("--game", metavar="SLUG",
                        help="download one catalogued transcript into transcripts/")
    parser.add_argument("--url", metavar="URL",
                        help="install a transcript straight from a supported page "
                             "URL (no catalog entry needed); prints the slug")
    parser.add_argument("--force", action="store_true",
                        help="redownload transcript files even if they already exist")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.all, args.game, args.url)) > 1:
        parser.error("choose only one of --all, --game, --url")

    if not args.all and not args.game and not args.url:
        parser.error("choose --game SLUG, --url URL, or --all")

    if args.url:
        try:
            slug, path = download_from_url(args.url, force=args.force)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {path.relative_to(BASE_DIR)}", file=sys.stderr)
        print(slug)
        return 0

    catalog = load_catalog()
    if args.game:
        path = download_game(args.game, catalog=catalog, force=args.force)
        print(f"wrote {path.relative_to(BASE_DIR)}")
        return 0
    return build_all(catalog, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
