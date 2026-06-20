# private_docs

`private_docs/` is the internal working record for this research repository.
It separates the current source of truth from longer design notes, experiment
records, and historical material.

Read in this order:

1. `CURRENT_STATE.md`
2. `IMPLEMENTATION_SCHEDULE.md`
3. `NEXT_CHAT_HANDOFF.md`
4. `reference/INDEX.md`
5. `experiments/INDEX.md`

`CURRENT_STATE.md` is authoritative for the active mainline. Files under
`archive/` are historical and must not be treated as current unless an active
document explicitly points to them.

Development follows one request per theme, strict validation, minimal patches,
and relevant tests before the full suite. Temporary compatibility shims are not
added unless they are explicitly documented with a removal condition.
