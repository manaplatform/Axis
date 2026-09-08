# Axis

**AI Agent for Cloud, Kubernetes & Docker Operations**

Axis is an open-source AI agent designed to help you understand, diagnose, and safely operate modern infrastructure. It focuses on **Cloud + Kubernetes + Docker** environments with strong safety controls and human-in-the-loop approvals.

## Features (MVP)

- **Status overview** of clusters, nodes, pods, and containers
- **Intelligent diagnosis** of services and failures
- **Log collection & summarization**
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
- [ ] Core CLI & configuration
- [ ] Kubernetes tools
- [ ] Docker tools
- [ ] Cloud CLI integrations
- [ ] Analyzer + Planner agents
- [ ] Safety & permission system
- [ ] GitOps PR generation
- [ ] Dashboard (future)

## License

MIT
