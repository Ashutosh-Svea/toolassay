"""Invented bookshop data, generated deterministically so every run sees the same shelves."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

SECTIONS = ("Poetry", "Fiction", "History", "Science", "Cooking", "Travel")

_ADJECTIVES = (
    "Quiet",
    "Amber",
    "Hollow",
    "Winter",
    "Copper",
    "Paper",
    "Long",
    "Blue",
    "Wandering",
    "Iron",
    "Green",
    "Last",
    "Little",
    "Restless",
    "Distant",
    "Crooked",
    "Bright",
    "Salt",
)
_NOUNS = (
    "Harbour",
    "Orchard",
    "Meridian",
    "Compass",
    "Garden",
    "Ledger",
    "Tide",
    "Kitchen",
    "Archive",
    "Road",
    "Furnace",
    "Almanac",
    "Atlas",
    "Field",
    "Bridge",
    "Lantern",
    "Signal",
)
_FIRST_NAMES = (
    "Teodor",
    "Ines",
    "Odile",
    "Ravi",
    "Sunniva",
    "Kwame",
    "Halle",
    "Piet",
    "Noor",
    "Ezra",
    "Lotte",
    "Mara",
)
_LAST_NAMES = (
    "Hale",
    "Marlowe",
    "Brandt",
    "Sefton",
    "Okafor",
    "Lindqvist",
    "Abara",
    "Vance",
    "Duarte",
    "Ferro",
    "Kaspar",
    "Quill",
)
_PUBLISHERS = ("Larkspur Press", "Tidewater Books", "Northlight House", "Mill and Meadow")

CATALOGUE_SIZE = 200
SEED = 7


@dataclass(frozen=True)
class Book:
    book_id: str
    title: str
    author: str
    section: str
    price: float
    stock: int
    shelf: str
    isbn: str
    publisher: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# Books the example cases refer to. Their values are fixed so the case file can assert on them.
FEATURED_BOOKS: tuple[Book, ...] = (
    Book(
        "BK-0007",
        "Salt Roads",
        "Ines Marlowe",
        "Travel",
        12.50,
        3,
        "T1",
        "978-1-402011-13-6",
        "Tidewater Books",
    ),
    Book(
        "BK-0042",
        "The Quiet Lantern",
        "Mara Quill",
        "Poetry",
        11.00,
        7,
        "P3",
        "978-1-517330-42-0",
        "Larkspur Press",
    ),
    Book(
        "BK-0043",
        "Salt and Ember",
        "Mara Quill",
        "Poetry",
        6.50,
        2,
        "P3",
        "978-1-517330-43-7",
        "Larkspur Press",
    ),
    Book(
        "BK-0051",
        "Orchard Arithmetic",
        "Teodor Hale",
        "Science",
        18.00,
        0,
        "S2",
        "978-1-640221-51-9",
        "Northlight House",
    ),
    Book(
        "BK-0088",
        "Nine Winters",
        "Odile Brandt",
        "Fiction",
        14.00,
        5,
        "F4",
        "978-1-733901-88-2",
        "Mill and Meadow",
    ),
)

_FEATURED_TITLES = {book.title for book in FEATURED_BOOKS}
_FEATURED_AUTHORS = {book.author for book in FEATURED_BOOKS}


def build_catalogue() -> list[Book]:
    """Two hundred invented books; the featured ones replace generated entries by id."""
    rng = random.Random(SEED)
    featured_by_id = {book.book_id: book for book in FEATURED_BOOKS}
    used_titles = set(_FEATURED_TITLES)
    books: list[Book] = []
    for index in range(1, CATALOGUE_SIZE + 1):
        book_id = f"BK-{index:04d}"
        if book_id in featured_by_id:
            books.append(featured_by_id[book_id])
            continue
        title = _unique_title(rng, used_titles)
        author = _generated_author(rng)
        section = rng.choice(SECTIONS)
        books.append(
            Book(
                book_id=book_id,
                title=title,
                author=author,
                section=section,
                price=round(rng.uniform(9.0, 32.0), 2),
                stock=rng.randint(0, 12),
                shelf=f"{section[0]}{rng.randint(1, 6)}",
                isbn=_fake_isbn(rng),
                publisher=rng.choice(_PUBLISHERS),
            )
        )
    return books


def _fake_isbn(rng: random.Random) -> str:
    return f"978-1-{rng.randint(100000, 999999)}-{rng.randint(10, 99)}-{rng.randint(0, 9)}"


def _unique_title(rng: random.Random, used: set[str]) -> str:
    while True:
        title = f"{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)}"
        if rng.random() < 0.5:
            title = f"The {title}"
        if title not in used:
            used.add(title)
            return title


def _generated_author(rng: random.Random) -> str:
    while True:
        author = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
        if author not in _FEATURED_AUTHORS:
            return author
