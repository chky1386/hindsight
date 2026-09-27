"""A mistyped edit must not make a partly stale document look refreshed (#4829).

Drive retain, the worker's delta refresh, and its correction through the client.
The batch updates current housing and adds its history. If only history lands,
the page contradicts itself; if the watermark moves, the correction is lost.
"""

from __future__ import annotations

import re
from typing import Literal

import pytest
from pydantic import BaseModel

from hindsight_system_tests import reflect_loop, wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.rulebook import ChatRequest

pytestmark = pytest.mark.asyncio

BASELINE = "# Housing\n\nAlice lives in Berlin.\n"
CURRENT = "Alice lives in Paris."
HISTORY = "Alice moved from Berlin to Paris."


class AppendBlock(BaseModel):
    op: Literal["append_block"] = "append_block"
    section_id: str = "housing"
    text: str = HISTORY


class ReplaceBlock(BaseModel):
    op: Literal["replace_block"] = "replace_block"
    section_id: str = "housing"
    block_id: str
    text: str = CURRENT


class DeltaReply(BaseModel):
    operations: list[AppendBlock | ReplaceBlock]


@pytest.mark.parametrize("repair_on_retry", [True, False], ids=["corrected", "still_invalid"])
async def test_partial_delta_retries_before_writing(client, llm, bank_id, settled, repair_on_retry: bool):
    llm.on_step("extract_facts").returns(extracted(fact("Alice lives in Berlin", who="Alice", entities=["Alice"])))
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=BASELINE, query="Alice Berlin Paris")
    await client.aretain(bank_id=bank_id, content="Alice lives in Berlin.")
    await settled(bank_id)
    created = await client.mental_models.create_mental_model(
        bank_id, {"name": "Housing", "source_query": "Where does Alice live?", "trigger": {"mode": "delta"}}
    )
    await settled(bank_id)
    model_id = created.mental_model_id
    before = await client.mental_models.get_mental_model(bank_id, model_id, detail="full")
    assert before.content == BASELINE

    llm.reset()
    llm.on_step("extract_facts").returns(extracted(fact(HISTORY, who="Alice", entities=["Alice"])))
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=CURRENT, query="Alice Berlin Paris")
    await client.aretain(bank_id=bank_id, content=HISTORY)
    await settled(bank_id)

    calls: list[ChatRequest] = []

    def delta_reply(request: ChatRequest) -> DeltaReply:
        calls.append(request)
        # The server minted this opaque id. Read it from the document shown to
        # the provider, then reproduce the one-character truncation in #4829.
        block = re.search(r'"id":\s*"(b[0-9a-f]{8})"', request.user_text)
        assert block is not None
        block_id = block.group(1)
        if len(calls) == 1 or not repair_on_retry:
            block_id = block_id[:-1]
        return DeltaReply(operations=[AppendBlock(), ReplaceBlock(block_id=block_id)])

    llm.on_step("delta_ops").answers_with(delta_reply)
    submitted = await client.mental_models.refresh_mental_model(bank_id, model_id)
    await wait_until_settled(client, bank_id, allow_failed=not repair_on_retry)

    assert len(calls) == 2
    after = await client.mental_models.get_mental_model(bank_id, model_id, detail="full")
    operation = await client.operations.get_operation_status(bank_id, submitted.operation_id)
    assert operation.details is not None
    if repair_on_retry:
        assert after.content == f"# Housing\n\n{CURRENT}\n\n{HISTORY}\n"
        assert after.is_stale is False
        assert operation.status == "completed"
        assert operation.details.outcome == "content_written"
        assert after.reflect_response["delta_operations_skipped"] == []
    else:
        assert after.content == before.content
        assert after.last_refreshed_at == before.last_refreshed_at
        assert after.last_memory_seen_at == before.last_memory_seen_at
        assert after.is_stale is True
        assert operation.status == "failed"
        assert operation.details.outcome == "refresh_failed_delta_not_applied"
        assert operation.details.failure_reason == "delta_ops_failed"
        assert after.reflect_response["delta_applied"] is False
        skipped = after.reflect_response["delta_operations_skipped"]
        assert len(skipped) == 1
        assert skipped[0]["reason"].startswith("unknown block_id:")

        # No new retain: the failed edit's evidence must still be inside the
        # next delta window. A manual refresh can now recover the whole batch.
        repair_on_retry = True
        recovered = await client.mental_models.refresh_mental_model(bank_id, model_id)
        await wait_until_settled(client, bank_id, allow_failed=True)
        assert len(calls) == 3
        current = await client.mental_models.get_mental_model(bank_id, model_id, detail="full")
        assert current.content == f"# Housing\n\n{CURRENT}\n\n{HISTORY}\n"
        assert current.is_stale is False
        assert (await client.operations.get_operation_status(bank_id, recovered.operation_id)).status == "completed"
