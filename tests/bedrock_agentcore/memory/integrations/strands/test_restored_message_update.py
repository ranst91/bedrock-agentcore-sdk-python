"""Regression coverage for updates using restored Strands logical message IDs."""

import copy
import json
from datetime import datetime, timezone

import pytest
from strands.types.exceptions import SessionException
from strands.types.session import SessionMessage

from bedrock_agentcore.memory.integrations.strands.bedrock_converter import AgentCoreMemoryConverter

from . import test_agentcore_memory_session_manager as fixtures

agentcore_config = fixtures.agentcore_config
mock_memory_client = fixtures.mock_memory_client
_create_session_manager = fixtures._create_session_manager


def make_message(text, created_at):
    return SessionMessage(message={"role": "user", "content": [{"text": text}]}, message_id=0, created_at=created_at)


def payload(message, blob=False):
    text, role = AgentCoreMemoryConverter.message_to_payload(message)[0]
    return (
        {"blob": json.dumps([text, role])}
        if blob
        else {"conversational": {"content": {"text": text}, "role": role.upper()}}
    )


@pytest.mark.parametrize("blob", [False, True])
def test_restored_update_preserves_identity_order_and_siblings(agentcore_config, mock_memory_client, blob):
    earlier = make_message("previous", "2026-10-13T10:00:00+00:00")
    original = make_message("Forwarded to client", "2026-10-13T10:01:00+00:00")
    later = make_message("following", "2026-10-13T10:02:00+00:00")
    timestamp = datetime(2026, 10, 13, 10, 3, tzinfo=timezone.utc)
    event = {
        "eventId": "123#abc",
        "eventTimestamp": timestamp,
        "payload": [payload(earlier), payload(original, blob), payload(later)],
        "metadata": {"custom": {"stringValue": "keep"}},
    }
    events = [copy.deepcopy(event)]
    mock_memory_client.list_events.side_effect = lambda **kwargs: copy.deepcopy(events)

    def create(**kwargs):
        replacement = {**copy.deepcopy(kwargs), "eventId": "123#def"}
        events.append(replacement)
        return {"event": replacement}

    def delete(**kwargs):
        assert isinstance(kwargs["eventId"], str)
        events[:] = [item for item in events if item["eventId"] != kwargs["eventId"]]

    manager = _create_session_manager(agentcore_config, mock_memory_client)
    mock_memory_client.gmdp_client.create_event.side_effect = create
    mock_memory_client.gmdp_client.delete_event.side_effect = delete
    restored = manager.list_messages(agentcore_config.session_id, "agent")
    restored[1].message["content"] = [{"text": "actual correlated answer"}]
    manager.update_message(agentcore_config.session_id, "agent", restored[1])

    fresh = _create_session_manager(agentcore_config, mock_memory_client)
    actual = fresh.list_messages(agentcore_config.session_id, "agent")
    assert [m.to_dict() for m in actual] == [m.to_dict() for m in restored]
    assert len(events) == 1
    assert events[0]["eventTimestamp"] == timestamp
    assert events[0]["metadata"] == event["metadata"]
    assert events[0]["payload"][0] == event["payload"][0]
    assert events[0]["payload"][2] == event["payload"][2]


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_ambiguous_identity_does_not_mutate(agentcore_config, mock_memory_client, count):
    message = make_message("answer", "2026-10-13T10:01:00+00:00")
    mock_memory_client.list_events.return_value = [
        {"eventId": f"123#abc{i}", "payload": [payload(message)]} for i in range(count)
    ]
    manager = _create_session_manager(agentcore_config, mock_memory_client)
    with pytest.raises(SessionException, match="uniquely"):
        manager.update_message(agentcore_config.session_id, "agent", message)
    mock_memory_client.create_event.assert_not_called()
    mock_memory_client.gmdp_client.create_event.assert_not_called()
    mock_memory_client.gmdp_client.delete_event.assert_not_called()


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_restored_update_rolls_back_on_delete_failure(agentcore_config, mock_memory_client, rollback_fails):
    message = make_message("answer", "2026-10-13T10:01:00+00:00")
    mock_memory_client.list_events.return_value = [
        {
            "eventId": "123#abc",
            "eventTimestamp": datetime(2026, 10, 13, tzinfo=timezone.utc),
            "payload": [payload(message)],
        }
    ]
    manager = _create_session_manager(agentcore_config, mock_memory_client)
    mock_memory_client.gmdp_client.create_event.return_value = {"event": {"eventId": "123#def"}}
    mock_memory_client.gmdp_client.delete_event.side_effect = [
        RuntimeError("delete failed"),
        RuntimeError("rollback failed") if rollback_fails else None,
    ]
    with pytest.raises(SessionException, match="both events" if rollback_fails else "delete failed"):
        manager.update_message(agentcore_config.session_id, "agent", message)
    assert [c.kwargs["eventId"] for c in mock_memory_client.gmdp_client.delete_event.call_args_list] == [
        "123#abc",
        "123#def",
    ]
