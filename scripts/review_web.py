"""Check the in-room Scout labels in a browser: each photo with its draft boxes, the
keys on screen, and a warning when the session's prop has no box.

    uv run scripts/review_web.py
    open http://127.0.0.1:8020            (space = boxes are right, next photo)

Same files and the same reviewed.txt as review_labels.py, so build and train need
nothing new. Test photos come first: build refuses any that are unreviewed.
Localhost only: the photos show whoever was in the room."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from review_labels import mark_reviewed, prop_default_class
from scout_data import (
    CLASSES,
    PROP_CLASS,
    load_reviewed,
    prop_of,
    read_labels,
    room_images,
    room_splits,
    write_labels,
)

PAGE = Path(__file__).with_name("review_web.html")
FONTS = ROOT / "sherlock" / "sherlock" / "dashboard" / "static" / "fonts"
SCALE = 1000  # boxes travel as x1 y1 x2 y2 on 0-1000, the page's canvas units
SPLIT_ORDER = ("test", "val", "train")


class Box(BaseModel):
    cls: int = Field(ge=0, lt=len(CLASSES))
    box: tuple[float, float, float, float]


class Labels(BaseModel):
    labels: list[Box]


def make_app(props: Path, cache: Path) -> FastAPI:
    app = FastAPI()

    def photos() -> dict[str, Path]:
        """Every in-room photo by its path under props, test split first."""
        imgs = room_images(props)
        splits = room_splits(imgs)
        imgs.sort(key=lambda p: (SPLIT_ORDER.index(splits[p]), str(p)))
        return {str(p.relative_to(props)): p for p in imgs}

    def photo(rel: str) -> Path:
        path = photos().get(rel)  # only known photos: no path tricks reach the disk
        if path is None:
            raise HTTPException(404)
        return path

    def drafted(path: Path) -> bool:
        return (cache / path.relative_to(props)).with_suffix(".json").exists()

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text()

    @app.get("/fonts/{name}")
    def font(name: str) -> FileResponse:
        path = FONTS / name
        if path.parent != FONTS or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/api/photos")
    def listing() -> dict:
        all_ = photos()
        splits = room_splits(all_.values())
        done = load_reviewed(props)
        return {
            "classes": list(CLASSES),
            "photos": [{"rel": rel, "split": splits[p], "prop": prop_of(p), "reviewed": rel in done,
                        "gemini": drafted(p)} for rel, p in all_.items()],
        }

    @app.get("/api/labels/{rel:path}")
    def labels(rel: str) -> dict:
        path = photo(rel)
        want = PROP_CLASS.get(prop_of(path))
        return {
            "labels": [{"cls": c, "box": b} for c, b in read_labels(path.with_suffix(".txt"), SCALE, SCALE)],
            "default_class": CLASSES.index(prop_default_class(rel)),
            "prop_class": CLASSES.index(want) if want else None,
            "reviewed": rel in load_reviewed(props),
            "gemini": drafted(path),
        }

    @app.post("/api/labels/{rel:path}")
    def save(rel: str, body: Labels) -> dict:
        path = photo(rel)
        write_labels(path.with_suffix(".txt"), [(b.cls, b.box) for b in body.labels], SCALE, SCALE)
        mark_reviewed(props, rel)
        return {"reviewed": len(load_reviewed(props))}

    @app.get("/photos/{rel:path}")
    def image(rel: str) -> FileResponse:
        return FileResponse(photo(rel))

    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--props", type=Path, default=ROOT / "data" / "props")
    p.add_argument("--cache", type=Path, help="Gemini answers (default: <props>/../gemini_labels)")
    p.add_argument("--port", type=int, default=8020)
    a = p.parse_args(argv)
    print(f"review page: http://127.0.0.1:{a.port}  (Ctrl-C to stop)")
    uvicorn.run(make_app(a.props, a.cache or a.props.parent / "gemini_labels"),
                host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
