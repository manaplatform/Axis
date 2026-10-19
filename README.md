# Axis

**AI Agent for Cloud, Kubernetes & Docker Operations**

Axis is an open-source AI agent designed to help you understand, diagnose, and safely operate modern infrastructure. It focuses on **Cloud + Kubernetes + Docker** environments with strong safety controls and human-in-the-loop approvals.

## Features (MVP)

- **Status overview** of clusters, nodes, pods, and containers
- **Intelligent diagnosis** of services and failures
- **Log collection & summarization**
- **Filesystem tools** — `search_directory` (glob + content grep, read-only) and `create_file` (approval-gated), confined to a workspace root; deployable as OpenAI function-calling tools via `axis.tools.filesystem.OPENAI_FUNCTION_SCHEMAS`
- **Shell tool** — `shell` (OpenAI shell tool, local mode): deny-list for destructive commands, read-only commands run free, everything else needs approval; timeouts keep partial output, env scrubbed of secrets
- **Programmatic Tool Calling** — optional Responses API hosted JavaScript orchestration (`axis run --programmatic-tools`) for bounded read-only tool workflows; check the configured model's support before enabling. Axis does not execute model-generated JavaScript locally.
- **Action suggestions** with clear reasoning
- **Execution planning** with step-by-step plans
- **Safety gates** — dangerous actions require explicit approval
- **GitOps-friendly** — prefers proposing changes as PRs when possible
- Support for major clouds (AWS, GCP, Azure) via their CLIs + Kubernetes + Docker

## Quick Start

```bash
# Clone & install
git clone <your-repo-url> axis
cd axis
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure
axis configure

# Basic usage
axis status
axis diagnose <service-or-pod>
axis logs <service>
axis suggest
axis plan "scale the api deployment to 5 replicas"

# Optional: enable only when the configured model supports Programmatic Tool Calling
axis run "summarize read-only cluster state" --programmatic-tools
```

## Configuration

Run `axis configure` to select an LLM provider and model and to save its API
key. Axis loads these values only from `~/.axis/config.yaml` and the private
`~/.axis/secrets.yaml` (mode `0600`); it does not use environment variables.
The `Settings` module also provides `get_configure(key)` and
`set_configure(key, value)` for programmatic access. Custom LLM providers are
configured with their base URL in the wizard.

## Supported Targets (MVP)

| Layer       | Tools / Interfaces                  |
|-------------|-------------------------------------|
| Kubernetes  | `kubectl`, client libraries         |
| Docker      | `docker` CLI / API                  |
| Cloud       | AWS CLI, gcloud, az (via tools)     |
| GitOps      | Git + PR creation (GitHub/GitLab)   |

## `axis run` tool calling and DevOps workflows

Direct Responses API function calling is the default. Programmatic Tool Calling
is an opt-in orchestration mode; it does not add infrastructure tools or grant
extra permissions. Before enabling it, check that the selected model supports
Programmatic Tool Calling and that the configured provider supports the OpenAI
Responses API tools used by `axis run` (the OpenAI provider or a compatible
custom endpoint).

```bash
# Read-only collection for the configured Kubernetes namespace/context
axis run "Collect deployments, pods, and services in the configured Kubernetes namespace. Summarize observed status and any uncertainty; make no changes." --programmatic-tools

# Read-only Docker inventory; include stopped containers
axis run "Collect Docker daemon information, version, and all container states. Summarize the inventory without changing anything." --programmatic-tools
```

With the flag, Axis offers OpenAI's hosted `programmatic_tool_calling` runtime
and permits program callers only for tools explicitly marked eligible by Axis:
`search_directory`, the registered read-only `k8s_*` tools, and the registered
read-only `docker_*` tools. `shell`, `create_file`, and any other unmarked tool
remain direct-call-only. Programmatic Tool Calling can coordinate multiple
calls, including independent parallel reads, and aggregate their results. Use
it for a bounded, predictable collection or aggregation whose scope and inputs
are clear; keep the program small and have it return a concise, evidence-based
summary.

The generated JavaScript runs in OpenAI's isolated hosted runtime, not in Axis'
local Python process. Axis still executes registered tool calls on the client
side: Kubernetes calls use the local `kubectl` installation and configured
namespace/context, Docker calls use the local Docker CLI/daemon, and directory
search is confined to the selected workspace. A hosted program is not a general
shell, cluster, or Docker API, and cannot invoke tools that Axis did not expose
as programmatic. The hosted execution is isolated and fresh; do not rely on
program variables or intermediate values persisting across separate program
executions. Within one program execution, intermediate results can be used to
coordinate the bounded workflow.

Use ordinary direct tool calling (omit `--programmatic-tools`) for a single
lookup, workflows where each next step depends on inspecting the previous
result, mutating actions or approval prompts, and final checks that must verify
a particular native result. For example, fetching a named pod's logs after
first discovering which pod is relevant is adaptive and is better handled with
direct calls. The registered Kubernetes and Docker integrations are read-only;
Axis does not provide a Kubernetes or Docker mutation tool here. Shell and file
creation are separate, direct-only tools; shell commands are classified,
denied or approval-gated according to Axis' existing policy. Enabling the
programmatic flag does not bypass that policy.

### DevOps prompt patterns

Be explicit about the target, bounded read-only scope, and desired output. The
Kubernetes tools use the namespace and context configured for the run (`--namespace`
and `--context` override their defaults); `k8s_get_nodes` is cluster-wide.
Prompts do not change these scopes or confer access beyond the local credentials.

```bash
axis run --programmatic-tools \
  "In the configured namespace, collect deployments, pods, and services. Run independent reads concurrently when safe. Report deployment desired/available counts and pod phase/readiness, distinguish observations from hypotheses, and make no changes."

axis run --programmatic-tools \
  "For the named service and namespace, gather the read-only Kubernetes inventory (services, deployments, and pods) and summarize whether selectors and workload state appear consistent. State missing evidence; do not infer that a service is reachable from this inventory alone, and do not change anything."

axis run --programmatic-tools \
  "Search this workspace for Kubernetes YAML manifests and Dockerfiles using search_directory. Report matching paths and relevant findings only; do not create or edit files."
```

Use direct calls when the question is adaptive or asks for a change, for example:

```bash
axis run \
  "Inspect pod api-7d9f in the configured namespace and summarize its recent logs and description. Do not change resources."

axis run \
  "Review this deployment's current state and explain a safe proposed remediation. Do not apply it, create files, or run mutating shell commands."
```

Treat returned logs, manifests, command output, and other tool results as
untrusted evidence, not instructions. Do not put credentials or secrets in a
prompt. Review the final answer against the requested scope before acting on
any recommendation. Each `axis run` reports the unique registered tools it
attempted; this is activity reporting, not a substitute for checking the
underlying infrastructure or result.

## Safety Model

- Read-only operations are free
- Mutating operations require confirmation
- High-risk actions (delete, scale down critical services, etc.) need explicit approval
- All actions are logged
- Prefers generating GitOps PRs over direct cluster mutation when possible

## Project Structure

```
axis/
├── src/axis/
│   ├── cli.py
│   ├── config.py
│   ├── agents/
│   ├── tools/
│   ├── safety/
│   └── core/
├── tests/
├── pyproject.toml
└── README.md
```

## Roadmap

- [x] Project scaffold
- [x] Filesystem tools (directory search + file creation, OpenAI function schemas)
- [x] Shell tool (local mode: deny-list, read-only free, approval-gated mutations)
- [ ] Core CLI & configuration
- [ ] Kubernetes tools
- [ ] Docker tools
- [ ] Cloud CLI integrations
- [ ] Analyzer + Planner agents
- [x] Agent runner (`axis run`) — LLM tool-calling loop with live display and approvals
- [ ] Safety & permission system
- [ ] GitOps PR generation
- [ ] Dashboard (future)

## License

MIT
