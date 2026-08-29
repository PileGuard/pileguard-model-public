"""Safely stream AI Hub laying-hen validation archives without extracting them."""

from __future__ import annotations

import io
import json
import math
import re
import tarfile
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from pileguard.features.nestler import BoundingBox
from pileguard.features.pio import NormalizedBox

STAGE_METADATA = {
    "early": {
        "filename_token": "산란초기",
        "life_cycle": "early-stage-of-egg-laying",
    },
    "middle": {
        "filename_token": "산란중기",
        "life_cycle": "middle-stage-of-egg-production",
    },
    "late": {
        "filename_token": "산란후기",
        "life_cycle": "after-laying-eggs",
    },
}
PART_PATTERN = re.compile(r"^(?P<base>.+)\.part(?P<offset>\d+)$")


@dataclass(frozen=True)
class AihubArchivePair:
    stage: str
    label_archive: Path
    source_archive: Path


@dataclass(frozen=True)
class AihubAnnotation:
    image_id: str
    stage: str
    width: int
    height: int
    pixel_boxes: tuple[BoundingBox, ...]
    normalized_boxes: tuple[NormalizedBox, ...]
    action_positive_boxes: int
    clipped_boxes: int
    invalid_boxes: int = 0


def normalized_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def validate_archive_name(name: str) -> PurePosixPath:
    """Reject paths and separators that could escape an extraction root."""

    normalized = normalized_text(name)
    if "\\" in normalized:
        raise ValueError(f"Archive member uses a backslash separator: {name!r}")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return path


def discover_archive_pairs(archive_dir: Path) -> list[AihubArchivePair]:
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"AI Hub archive directory not found: {archive_dir}")
    tar_paths = [path for path in archive_dir.iterdir() if path.is_file() and path.suffix == ".tar"]
    pairs: list[AihubArchivePair] = []
    for stage, metadata in STAGE_METADATA.items():
        token = str(metadata["filename_token"])
        stage_paths = [path for path in tar_paths if token in normalized_text(path.name)]
        labels = [path for path in stage_paths if "라벨" in normalized_text(path.name)]
        sources = [path for path in stage_paths if "원천" in normalized_text(path.name)]
        if len(labels) != 1 or len(sources) != 1:
            raise ValueError(
                f"Expected one label and one source TAR for {stage}; "
                f"labels={len(labels)} sources={len(sources)}"
            )
        pairs.append(AihubArchivePair(stage, labels[0], sources[0]))
    return pairs


class MultipartReader(io.RawIOBase):
    """Expose AI Hub offset-named TAR members as one sequential binary stream."""

    def __init__(self, archive: tarfile.TarFile, parts: list[tarfile.TarInfo]) -> None:
        super().__init__()
        self._archive = archive
        self._parts = iter(parts)
        self._current: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        target = memoryview(buffer).cast("B")
        written = 0
        while written < len(target):
            if self._current is None:
                try:
                    member = next(self._parts)
                except StopIteration:
                    break
                extracted = self._archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Could not read split archive member: {member.name}")
                self._current = extracted
            count = self._current.readinto(target[written:])
            if count:
                written += count
                continue
            self._current.close()
            self._current = None
        return written

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None
        super().close()


def split_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    parts: list[tuple[int, tarfile.TarInfo]] = []
    common_base: str | None = None
    for member in archive.getmembers():
        validate_archive_name(member.name)
        if member.isdir():
            continue
        if not member.isfile():
            raise ValueError(f"AI Hub wrapper contains a non-regular member: {member.name}")
        match = PART_PATTERN.fullmatch(normalized_text(PurePosixPath(member.name).name))
        if match is None:
            raise ValueError(f"Unexpected AI Hub split member name: {member.name}")
        base = match.group("base")
        if common_base is None:
            common_base = base
        elif base != common_base:
            raise ValueError("AI Hub wrapper contains parts for more than one inner archive")
        parts.append((int(match.group("offset")), member))
    if not parts:
        raise ValueError("AI Hub wrapper contains no split archive parts")
    parts.sort(key=lambda item: item[0])
    expected_offset = 0
    for offset, member in parts:
        if offset != expected_offset:
            raise ValueError(
                "AI Hub split archive is incomplete: "
                f"expected offset {expected_offset}, got {offset}"
            )
        expected_offset += member.size
    return [member for _, member in parts]


@contextmanager
def open_inner_tar(path: Path) -> Iterator[tarfile.TarFile]:
    """Open the gzip TAR reconstructed from one official AI Hub wrapper TAR."""

    with tarfile.open(path, mode="r:") as outer:
        reader = MultipartReader(outer, split_members(outer))
        buffered = io.BufferedReader(reader, buffer_size=1024 * 1024)
        try:
            with tarfile.open(fileobj=buffered, mode="r|gz") as inner:
                yield inner
        finally:
            buffered.close()


def iter_inner_files(path: Path, *, suffix: str) -> Iterator[tuple[str, bytes]]:
    """Yield regular inner files with safe paths and a requested suffix."""

    seen: set[str] = set()
    with open_inner_tar(path) as inner:
        for member in inner:
            safe_path = validate_archive_name(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Inner archive contains a non-regular member: {member.name}")
            if safe_path.suffix.lower() != suffix.lower():
                continue
            basename = normalized_text(safe_path.name)
            if basename in seen:
                raise ValueError(f"Duplicate inner archive basename: {basename}")
            extracted = inner.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not read inner archive member: {member.name}")
            payload = extracted.read()
            if len(payload) != member.size:
                raise ValueError(f"Truncated inner archive member: {member.name}")
            seen.add(basename)
            yield basename, payload


def required_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected object at {field}")
    return value


def parse_annotation(payload: bytes, *, stage: str, member_name: str) -> AihubAnnotation:
    if stage not in STAGE_METADATA:
        raise ValueError(f"Unknown laying stage: {stage}")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Malformed AI Hub JSON: {member_name}") from error
    root = required_mapping(document, field="root")
    image = required_mapping(root.get("image"), field="image")
    annotation_info = required_mapping(
        root.get("annotationImageInfo"), field="annotationImageInfo"
    )
    objects = root.get("annotationObjectInfo")
    if not isinstance(objects, list):
        raise ValueError("Expected array at annotationObjectInfo")

    image_id = normalized_text(str(image.get("fileName", "")))
    validate_archive_name(image_id)
    if PurePosixPath(image_id).name != image_id or not image_id.lower().endswith(".png"):
        raise ValueError(f"Invalid AI Hub image filename: {image_id!r}")
    if normalized_text(Path(member_name).stem) != normalized_text(Path(image_id).stem):
        raise ValueError(f"Label/image stem mismatch: {member_name} vs {image_id}")
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image dimensions for {image_id}: {width}x{height}")
    expected_lifecycle = str(STAGE_METADATA[stage]["life_cycle"])
    if annotation_info.get("lifeCycle") != expected_lifecycle:
        raise ValueError(
            f"Unexpected lifeCycle for {stage}: {annotation_info.get('lifeCycle')!r}"
        )
    if annotation_info.get("action") != "community":
        raise ValueError(f"Expected community action for {image_id}")

    pixel_boxes: list[BoundingBox] = []
    normalized_boxes: list[NormalizedBox] = []
    action_positive = 0
    clipped = 0
    invalid = 0
    for index, value in enumerate(objects):
        item = required_mapping(value, field=f"annotationObjectInfo[{index}]")
        if item.get("category") != "layer-chicken":
            raise ValueError(f"Unexpected category in {image_id}: {item.get('category')!r}")
        action_value = item.get("actionValue")
        if not isinstance(action_value, bool):
            raise ValueError(f"Invalid actionValue in {image_id} at index {index}")
        action_positive += int(action_value)
        crowd_value = item.get("iscrowd", item.get("isCrowd"))
        if crowd_value not in (0, False):
            raise ValueError(f"Unexpected isCrowd value in {image_id} at index {index}")
        bbox = item.get("BBox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid BBox in {image_id} at index {index}")
        try:
            x, y, box_width, box_height = (float(number) for number in bbox)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Non-numeric BBox in {image_id} at index {index}") from error
        if not all(math.isfinite(number) for number in (x, y, box_width, box_height)):
            raise ValueError(f"Non-finite BBox in {image_id} at index {index}")
        if box_width <= 0 or box_height <= 0:
            invalid += 1
            continue
        x1 = max(x, 0.0)
        y1 = max(y, 0.0)
        x2 = min(x + box_width, float(width))
        y2 = min(y + box_height, float(height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"BBox outside image in {image_id} at index {index}")
        clipped += int((x1, y1, x2, y2) != (x, y, x + box_width, y + box_height))
        pixel_boxes.append(BoundingBox(x1, y1, x2, y2, index))
        normalized_boxes.append(
            NormalizedBox(
                class_id=0,
                center_x=((x1 + x2) / 2) / width,
                center_y=((y1 + y2) / 2) / height,
                width=(x2 - x1) / width,
                height=(y2 - y1) / height,
            )
        )
    return AihubAnnotation(
        image_id=image_id,
        stage=stage,
        width=width,
        height=height,
        pixel_boxes=tuple(pixel_boxes),
        normalized_boxes=tuple(normalized_boxes),
        action_positive_boxes=action_positive,
        clipped_boxes=clipped,
        invalid_boxes=invalid,
    )


def load_annotations(pair: AihubArchivePair) -> dict[str, AihubAnnotation]:
    annotations: dict[str, AihubAnnotation] = {}
    for member_name, payload in iter_inner_files(pair.label_archive, suffix=".json"):
        annotation = parse_annotation(payload, stage=pair.stage, member_name=member_name)
        if annotation.image_id in annotations:
            raise ValueError(f"Duplicate AI Hub image annotation: {annotation.image_id}")
        annotations[annotation.image_id] = annotation
    if not annotations:
        raise ValueError(f"No JSON labels found in {pair.label_archive.name}")
    return annotations


def dataset_audit(
    pairs: list[AihubArchivePair],
    *,
    split: str = "Validation",
) -> tuple[dict[str, dict[str, AihubAnnotation]], dict[str, Any]]:
    annotations_by_stage: dict[str, dict[str, AihubAnnotation]] = {}
    stages: dict[str, Any] = {}
    for pair in pairs:
        annotations = load_annotations(pair)
        annotations_by_stage[pair.stage] = annotations
        stages[pair.stage] = {
            "image_count": len(annotations),
            "box_count": sum(len(item.pixel_boxes) for item in annotations.values()),
            "action_positive_boxes": sum(
                item.action_positive_boxes for item in annotations.values()
            ),
            "clipped_boxes": sum(item.clipped_boxes for item in annotations.values()),
            "invalid_boxes": sum(item.invalid_boxes for item in annotations.values()),
            "resolutions": sorted(
                {
                    f"{item.width}x{item.height}"
                    for item in annotations.values()
                }
            ),
        }
    return annotations_by_stage, {
        "source": f"AI Hub dataset 575, official {split} community-image subset",
        "split": split,
        "stage_count": len(stages),
        "image_count": sum(stage["image_count"] for stage in stages.values()),
        "box_count": sum(stage["box_count"] for stage in stages.values()),
        "action_positive_boxes": sum(
            stage["action_positive_boxes"] for stage in stages.values()
        ),
        "invalid_boxes": sum(stage["invalid_boxes"] for stage in stages.values()),
        "stages": stages,
        "claim_boundary": (
            "Community images provide laying-hen boxes and static crowding scenes. They do not "
            "provide continuous timing or piling/smothering incident labels."
        ),
    }
