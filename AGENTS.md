# Agent instructions

All Axis configuration, especially LLM provider, model, endpoint, and API-key
values, must use the `axis.config` configuration module. Do not read, write, or
document environment variables as Axis configuration inputs. Use
`Settings.get`, `Settings.set`, `get_configure`, or `set_configure`; use the
`axis configure` wizard for interactive changes.

Every repository change must be recorded in `CHANGELOG.md` in the same change.
