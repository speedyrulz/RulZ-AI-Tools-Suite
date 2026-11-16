"""Inspect image and video datasets without loading training dependencies."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import os
import string
from PIL import Image, UnidentifiedImageError


@dataclass
class ImageInfo:
    path: Path
    width: int
    height: int
    mp: float
    caption: str | None


@dataclass
class DatasetScan:
    folders: list[Path]
    images: list[ImageInfo]
    videos: list[Path]
    captioned: int
    median_mp: float
    min_side: int
    aspect_buckets: dict[str, int]
    duplicates: list[tuple[Path, Path]]
    small: list[Path]
    trigger_found: str | None
    warnings: list[str]

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def has_captions(self) -> bool:
        # At least 80% of images must have caption files.
        return bool(self.images) and self.captioned / len(self.images) >= 0.8


def scan(folders: list[Path], trigger: str | None = None, caption_ext: str = '.txt') -> DatasetScan:
    images, videos, duplicates, small, warnings, hashes = [], [], [], [], [], {}
    seen = set()
    for folder in map(Path, folders):
        try:
            paths = []
            for root, dirs, files in os.walk(folder):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'excluded_by_look']
                paths.extend(Path(root) / name for name in files if not name.startswith('.'))
        except OSError as exc:
            warnings.append(f'{folder}: {exc}')
            continue
        for path in sorted(paths):
            if path in seen:
                continue
            seen.add(path)
            ext = path.suffix.lower()
            if ext in {'.mp4', '.mov', '.mkv', '.webm'}:
                videos.append(path)
            elif ext in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}:
                try:
                    with Image.open(path) as im:
                        width, height = im.size
                        thumb = im.convert('L').resize((8, 8))
                        pixels = list(thumb.tobytes())
                        avg = sum(pixels) / len(pixels)
                        ahash = sum((pixel >= avg) << i for i, pixel in enumerate(pixels))
                    caption_path = path.with_suffix(caption_ext)
                    caption = (caption_path.read_text(encoding='utf-8', errors='replace').strip() or None) if caption_path.is_file() else None
                    images.append(ImageInfo(path, width, height, width * height / 1_000_000, caption))
                    # Average hashes within four bits are near duplicates.
                    # Five disjoint chunks guarantee a shared chunk within four bit errors.
                    if max(pixels) - min(pixels) > 2:
                        candidates = set()
                        for chunk in range(5):
                            part = (ahash >> (chunk * 13)) & ((1 << (12 if chunk == 4 else 13)) - 1)
                            candidates.update(hashes.get((chunk, part), []))
                        for other, value in candidates:
                            if (ahash ^ value).bit_count() <= 4:
                                duplicates.append((other, path))
                        for chunk in range(5):
                            part = (ahash >> (chunk * 13)) & ((1 << (12 if chunk == 4 else 13)) - 1)
                            hashes.setdefault((chunk, part), []).append((path, ahash))
                    # The trainer's small-image warning starts below 512 pixels.
                    if min(width, height) < 512:
                        small.append(path)
                except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, UnidentifiedImageError) as exc:
                    warnings.append(f'{path}: {exc}')
    captions = [i.caption for i in images if i.caption is not None]
    stopwords = {'a', 'an', 'the', 'photo', 'this', 'portrait', 'image', 'of', 'picture', 'photograph'}
    tokens = Counter()
    for caption in captions:
        words = caption.split(',')[0].split()
        if words:
            token = words[0].lower().rstrip(string.punctuation)
            if token and token not in stopwords:
                tokens[token] += 1
    found = None
    if tokens:
        token, count = tokens.most_common(1)[0]
        # A trigger is considered common at 60% of caption files.
        if count / len(captions) >= 0.6:
            found = token
    buckets = Counter('square' if 0.9 <= i.width / i.height <= 1.1 else
                      'landscape' if i.width > i.height else 'portrait' for i in images)
    if not images and not videos:
        warnings.append('No images or videos found.')
    if small:
        warnings.append(f'{len(small)} images have a side below 512 pixels.')
    if duplicates:
        warnings.append(f'{len(duplicates)} possible duplicate pairs found.')
    if trigger and found and trigger.lower() != found:
        warnings.append(f'Caption trigger appears to be {found}, not {trigger}.')
    return DatasetScan(list(map(Path, folders)), images, videos, len(captions),
                       median([i.mp for i in images]) if images else 0.0,
                       min((min(i.width, i.height) for i in images), default=0), dict(buckets),
                       duplicates, small, found, warnings)
