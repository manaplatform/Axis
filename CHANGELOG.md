# Changelog

## Unreleased

- Added `axis run`: an LLM tool-calling agent loop over the OpenAI Responses
  API. The model picks tools from the new `axis.agents.registry`
  (name → local executor), the runner executes them locally with a live Rich
  display of each call and result, mutating tools are approved through the
  runner's permission gate before execution, and denials are reported back to
  the model. The loop ends at the final answer or `--max-steps`.
- Added `axis.agents.registry` with a `ToolRegistry` and `default_registry()`
  (filesystem tools first; kubernetes/docker/shell plug in with one call).

- Preserved the requested goal in LLM-generated plans so the CLI can render them.
- Switched the OpenAI provider to the official OpenAI Python client's Responses API.
- Retried custom OpenAI-compatible LLM requests without `response_format` when
  a provider rejects that optional parameter.
- Retried OpenAI requests without an unsupported optional `response_format`,
  allowing configured models that return JSON through ordinary chat content.
- Added HTTP status-only diagnostics for failed LLM provider requests.
- Returned the LLM failure reason with local planner fallback results.
- Replaced environment-variable configuration with file-backed configuration
  and removed the unused `python-dotenv` dependency.
- Added `Settings.get`/`Settings.set` and `get_configure`/`set_configure` APIs.
- Added custom LLM base-URL configuration to the interactive wizard.
- Added configuration guidance for agents and persistence/wizard tests.
