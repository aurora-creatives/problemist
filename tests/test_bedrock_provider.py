from unittest.mock import MagicMock, patch

import pytest

from problemist.providers.bedrock_provider import (
    AWSBedrockProvider,
    _translate_messages_bedrock,
    _translate_tools_bedrock,
)


def _provider(response: dict) -> tuple[AWSBedrockProvider, MagicMock]:
    with patch("problemist.providers.bedrock_provider.boto3.client") as mock_client:
        client = MagicMock()
        client.converse.return_value = response
        mock_client.return_value = client
        return AWSBedrockProvider(region="eu-west-1"), client


def test_translate_tools_to_converse_format():
    tools = [{"name": "search", "description": "find", "input_schema": {"type": "object", "properties": {}}}]
    spec = _translate_tools_bedrock(tools)[0]["toolSpec"]
    assert spec["name"] == "search"
    assert spec["inputSchema"]["json"]["type"] == "object"


def test_translate_messages_handles_text_tool_use_and_tool_result():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "looking"},
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    ]
    out = _translate_messages_bedrock(messages)
    assert out[0]["content"] == [{"text": "hi"}]
    assert out[1]["content"][1]["toolUse"]["toolUseId"] == "t1"
    assert out[2]["content"][0]["toolResult"]["content"] == [{"text": "ok"}]


@pytest.mark.asyncio
async def test_complete_returns_text_and_tool_calls():
    provider, client = _provider({
        "stopReason": "tool_use",
        "output": {"message": {"content": [
            {"text": "Checking."},
            {"toolUse": {"toolUseId": "t9", "name": "search", "input": {"q": "login"}}},
        ]}},
    })
    response = await provider.complete(
        [{"role": "user", "content": "go"}],
        tools=[{"name": "search"}],
        system="be brief",
    )
    assert response.provider == "aws_bedrock"
    assert response.stop_reason == "tool_use"
    assert response.content == "Checking."
    assert response.tool_calls[0].tool_name == "search"
    kwargs = client.converse.call_args.kwargs
    assert kwargs["system"] == [{"text": "be brief"}]
    assert "toolConfig" in kwargs


@pytest.mark.asyncio
async def test_unknown_stop_reason_falls_back_to_end_turn():
    provider, _ = _provider({"stopReason": "guardrail_intervened", "output": {"message": {"content": []}}})
    response = await provider.complete([{"role": "user", "content": "go"}])
    assert response.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_complete_simple_joins_text_blocks():
    provider, _ = _provider({"output": {"message": {"content": [{"text": "a"}, {"text": "b"}]}}})
    assert await provider.complete_simple("hello") == "a b"
