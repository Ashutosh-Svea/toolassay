from __future__ import annotations

import json

from toolassay.core import ToolCall
from toolassay.demo.catalogue import CATALOGUE_SIZE, FEATURED_BOOKS, build_catalogue
from toolassay.server import McpToolServer

from .support import connect


def test_catalogue_is_deterministic_and_well_formed() -> None:
    first = build_catalogue()
    second = build_catalogue()
    assert first == second
    assert len(first) == CATALOGUE_SIZE
    assert len({b.book_id for b in first}) == CATALOGUE_SIZE
    assert len({b.title for b in first}) == CATALOGUE_SIZE
    by_id = {b.book_id: b for b in first}
    for featured in FEATURED_BOOKS:
        assert by_id[featured.book_id] == featured
    poetry = sorted((b for b in first if b.section == "Poetry"), key=lambda b: b.price)
    assert poetry[0].title == "Salt and Ember"


async def _call(connection: McpToolServer, name: str, **arguments: object) -> str:
    result = await connection.call_tool(ToolCall(id="x", name=name, arguments=dict(arguments)))
    return result.text


async def test_v1_contracts_are_terse_and_unhelpful() -> None:
    async with connect("v1") as v1_connection:
        tools = {t.name: t for t in await v1_connection.list_tools()}
        assert tools["find_book"].description == "Find a book."
        assert "properties" in tools["list_inventory"].input_schema
        assert tools["list_inventory"].input_schema["properties"] == {}

        found = json.loads(await _call(v1_connection, "find_book", query="Mara Quill"))
        assert found and "book_id" not in found[0]
        assert await _call(v1_connection, "get_book", book_id="BK-9999") == "Error"
        everything = json.loads(await _call(v1_connection, "list_inventory"))
        assert len(everything) == CATALOGUE_SIZE
        bad_date = await _call(
            v1_connection,
            "reserve_book",
            book_id="BK-0088",
            customer_email="pat@example.com",
            pickup_date="12 September 2026",
        )
        assert bad_date == "Error: invalid request"


async def test_v2_contracts_explain_themselves() -> None:
    async with connect("v2") as v2_connection:
        tools = {t.name: t for t in await v2_connection.list_tools()}
        assert "book_id" in tools["find_book"].description
        assert "YYYY-MM-DD" in tools["reserve_book"].description

        found = json.loads(await _call(v2_connection, "find_book", query="Mara Quill"))
        assert {m["book_id"] for m in found["matches"]} == {"BK-0042", "BK-0043"}
        missing = json.loads(await _call(v2_connection, "find_book", query="Nowhere"))
        assert missing["matches"] == [] and "note" in missing
        assert "BK-0042" in await _call(v2_connection, "get_book", book_id="BK-9999")

        poetry = json.loads(await _call(v2_connection, "list_inventory", section="Poetry", limit=3))
        assert poetry["returned"] == 3
        assert poetry["rows"][0]["title"] == "Salt and Ember"
        assert poetry["total"] > 3
        assert "Sections are" in await _call(v2_connection, "list_inventory", section="Comics")

        bad_date = await _call(
            v2_connection,
            "reserve_book",
            book_id="BK-0088",
            customer_email="pat@example.com",
            pickup_date="12 September 2026",
        )
        assert "YYYY-MM-DD" in bad_date
        reserved = json.loads(
            await _call(
                v2_connection,
                "reserve_book",
                book_id="BK-0088",
                customer_email="pat@example.com",
                pickup_date="2026-09-12",
            )
        )
        assert reserved["reservation_id"].startswith("RSV-")
        assert "out of stock" in await _call(
            v2_connection,
            "reserve_book",
            book_id="BK-0051",
            customer_email="pat@example.com",
            pickup_date="2026-09-12",
        )
