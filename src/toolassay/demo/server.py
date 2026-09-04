"""A bookshop MCP server with two contract versions for the same four tools.

``v1`` is how tool contracts often ship: terse descriptions, results that omit the field the
next step needs, one tool that dumps the whole catalogue, and errors that say nothing.
``v2`` fixes the contracts without changing what the tools do. Run toolassay against both
and diff the runs to see what the contract alone is worth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from typing import Literal

from mcp.server.mcpserver import MCPServer

from toolassay.demo.catalogue import SECTIONS, Book, build_catalogue

Contracts = Literal["v1", "v2"]

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MAX_LIST_LIMIT = 50
_MAX_FIND_RESULTS = 5


class Bookshop:
    """Stateless operations over the catalogue; the tool layers only change their contracts."""

    def __init__(self) -> None:
        self.books = build_catalogue()
        self._by_id = {book.book_id: book for book in self.books}

    def find(self, query: str) -> list[Book]:
        needle = query.strip().lower()
        if not needle:
            return []
        return [
            book
            for book in self.books
            if needle in book.title.lower() or needle in book.author.lower()
        ][:_MAX_FIND_RESULTS]

    def get(self, book_id: str) -> Book | None:
        return self._by_id.get(book_id.strip().upper())

    def inventory(self, section: str | None = None) -> list[Book]:
        rows = self.books if section is None else [b for b in self.books if b.section == section]
        return sorted(rows, key=lambda book: (book.price, book.book_id))

    @staticmethod
    def reservation_id(book_id: str, email: str, pickup: str) -> str:
        digest = hashlib.sha256(f"{book_id}|{email}|{pickup}".encode()).hexdigest()
        return f"RSV-{digest[:6].upper()}"


def _compact(book: Book) -> dict[str, object]:
    return {
        "book_id": book.book_id,
        "title": book.title,
        "author": book.author,
        "section": book.section,
        "price": book.price,
        "stock": book.stock,
    }


def _parse_pickup_date(value: str) -> date | None:
    if not _ISO_DATE.match(value.strip()):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def build_server(contracts: Contracts) -> MCPServer:
    shop = Bookshop()
    server = MCPServer(
        "bookshop-demo",
        instructions="Catalogue and reservations for a small invented bookshop.",
        log_level="WARNING",
    )
    if contracts == "v1":
        _register_v1(server, shop)
    else:
        _register_v2(server, shop)
    return server


def _register_v1(server: MCPServer, shop: Bookshop) -> None:
    @server.tool(name="find_book", description="Find a book.", structured_output=False)
    def find_book(query: str) -> str:
        matches = shop.find(query)
        if not matches:
            return "No results."
        # Flaw: the result omits book_id, which get_book and reserve_book need.
        return json.dumps([{"title": b.title, "author": b.author} for b in matches])

    @server.tool(name="get_book", description="Get book.", structured_output=False)
    def get_book(book_id: str) -> str:
        book = shop.get(book_id)
        if book is None:
            return "Error"
        return json.dumps(book.as_dict())

    @server.tool(name="list_inventory", description="List inventory.", structured_output=False)
    def list_inventory() -> str:
        # Flaw: no way to narrow the result, so every call returns the whole catalogue.
        return json.dumps([b.as_dict() for b in shop.inventory()])

    @server.tool(name="reserve_book", description="Reserve a book.", structured_output=False)
    def reserve_book(book_id: str, customer_email: str, pickup_date: str) -> str:
        book = shop.get(book_id)
        pickup = _parse_pickup_date(pickup_date)
        # Flaw: every failure looks the same, so the caller cannot tell what to fix.
        if book is None or pickup is None or "@" not in customer_email or book.stock == 0:
            return "Error: invalid request"
        return json.dumps(
            {
                "reservation_id": shop.reservation_id(book.book_id, customer_email, pickup_date),
                "book_id": book.book_id,
                "pickup_date": pickup.isoformat(),
            }
        )


def _register_v2(server: MCPServer, shop: Bookshop) -> None:
    sections = ", ".join(SECTIONS)

    @server.tool(
        name="find_book",
        description=(
            "Search the catalogue by words from a title or an author's name (case-insensitive). "
            "Returns up to 5 matches, each with book_id, title, author, section, price and "
            "stock. Use get_book with a book_id when you also need the shelf location."
        ),
        structured_output=False,
    )
    def find_book(query: str) -> str:
        matches = shop.find(query)
        if not matches:
            return json.dumps(
                {
                    "matches": [],
                    "note": f"No title or author contains {query.strip()!r}. "
                    "Try fewer or different words.",
                }
            )
        return json.dumps({"matches": [_compact(b) for b in matches]})

    @server.tool(
        name="get_book",
        description=(
            "Fetch one book by its book_id (format BK-0000, as returned by find_book or "
            "list_inventory). Returns title, author, section, price, stock and shelf."
        ),
        structured_output=False,
    )
    def get_book(book_id: str) -> str:
        book = shop.get(book_id)
        if book is None:
            return (
                f"No book has book_id {book_id!r}. Ids look like BK-0042; "
                "use find_book to search by title or author."
            )
        return json.dumps({**_compact(book), "shelf": book.shelf})

    @server.tool(
        name="list_inventory",
        description=(
            f"List catalogue rows sorted by price ascending, optionally limited to one section "
            f"({sections}). Returns at most `limit` rows (default 20, max {_MAX_LIST_LIMIT}) "
            "plus the total count, so you can tell when rows were left out."
        ),
        structured_output=False,
    )
    def list_inventory(section: str | None = None, limit: int = 20) -> str:
        if section is not None and section not in SECTIONS:
            return f"Unknown section {section!r}. Sections are: {sections}."
        limit = max(1, min(limit, _MAX_LIST_LIMIT))
        rows = shop.inventory(section)
        return json.dumps(
            {
                "section": section or "all",
                "total": len(rows),
                "returned": min(limit, len(rows)),
                "rows": [_compact(b) for b in rows[:limit]],
            }
        )

    @server.tool(
        name="reserve_book",
        description=(
            "Reserve a book for in-store pickup. book_id is the BK-0000 id, customer_email is "
            "the customer's email address, and pickup_date must be an ISO date (YYYY-MM-DD), "
            "for example 2026-09-12. Returns a reservation_id, or a reason the reservation "
            "could not be made."
        ),
        structured_output=False,
    )
    def reserve_book(book_id: str, customer_email: str, pickup_date: str) -> str:
        book = shop.get(book_id)
        if book is None:
            return f"No book has book_id {book_id!r}. Use find_book to look up the id."
        if "@" not in customer_email:
            return f"customer_email {customer_email!r} is not an email address."
        pickup = _parse_pickup_date(pickup_date)
        if pickup is None:
            return (
                f"pickup_date {pickup_date!r} is not in YYYY-MM-DD form. "
                "Example: 2026-09-12. Convert the date and call reserve_book again."
            )
        if book.stock == 0:
            return f"{book.title} ({book.book_id}) is out of stock, so it cannot be reserved."
        return json.dumps(
            {
                "reservation_id": shop.reservation_id(book.book_id, customer_email, pickup_date),
                "book_id": book.book_id,
                "title": book.title,
                "pickup_date": pickup.isoformat(),
            }
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the bookshop demo MCP server.")
    parser.add_argument("--contracts", choices=("v1", "v2"), default="v1")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = build_server(args.contracts)
    if args.transport == "http":
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
