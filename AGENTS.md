# Generation collaboration rules

- Use tasks/collaboration.json and scripts/run_assignment.py for collaborator generation.
- Never alter a released assignment file or reuse its lot ID with different prompts.
- Before scheduling server jobs, exclude completed IDs, server-assigned IDs AND collaborator_reserved_ids. Check claims/* branches before reclaiming any lot.
- Do not release claim branches automatically on timeout. Confirm the previous worker stopped before a human-authorized transfer.
- Preserve prompt text, seed and model profile; prompt_id alone is not sufficient provenance across versions.
- Do not upload weights, environments, generated media, personal credentials or profiles/local.json.
- Do not bypass administrator job holds or run GPU inference on login nodes.
