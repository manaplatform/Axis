# Changelog

## Unreleased

- Added the `LICENSE` file with the MIT license text (copyright 2026 Axis
  Contributors), matching the license declared in `pyproject.toml` and the
  README.
- Merged the shell tool branch into the agent work: `axis.tools.shell`
  (`ShellTool.run`, `classify_command`) is now part of main-line history.
  `axis.tools` now exposes per-family schemas (`OPENAI_FUNCTION_SCHEMAS`
  stays filesystem-only for backwards compatibility; new
  `SHELL_FUNCTION_SCHEMAS` plus an `ALL_FUNCTION_SCHEMAS` aggregate).
- Wired the `shell` function into the agent runner registry: each command is
  classified per call — read-only commands run free, mutating ones need
  approval through the runner's gate, denied ones are blocked and reported
  to the model as an error.
- Wired the Kubernetes and Docker tools into the agent runner registry:
  `k8s_*` (get_pods, get_deployments, get_services, get_nodes, get_events,
  current_context, describe, logs) and `docker_*` (list_containers, logs,
  inspect, version, info) are now callable by the model through `axis run`.
  Both tool families are fully read-only, so they execute without approval;
  `axis run` gained `--namespace`/`--context` for the k8s tools.

## 0.0.2 — 2026-10-01

- Added `axis run`: an LLM tool-calling agent loop over the OpenAI Responses
  API. The model picks tools from the new `axis.agents.registry`
  (name → local executor), the runner executes them locally with a live Rich
  display of each call and result, mutating tools are approved through the
  runner's permission gate before execution, and denials are reported back to
  the model. The loop ends at the final answer or `--max-steps`.
- Added `axis.agents.registry` with a `ToolRegistry` and `default_registry()`
  (filesystem tools first; kubernetes/docker/shell plug in with one call).
- Fixed the agent loop echoing full SDK output items back into the Responses
  API input: read-only fields like `status` caused HTTP 400 "Unknown
  parameter". Only `function_call` items are echoed now, whitelisted to the
  fields the API accepts (`type`, `call_id`, `name`, `arguments`).

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
