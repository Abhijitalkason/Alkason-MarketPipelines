# ACTION ITEMS — Project Reset & Cleanup

> **Status:** PENDING — recorded 2026-07-24. Do **not** execute the deletions yet.
> **Sequencing:** These cleanup steps run **first**, but only *after* the new
> requirement is provided — because what counts as "old / not required" can only
> be decided against the new plan. Nothing is deleted until then.

---

## Context

The current solution (intraday system, v6 geometry, daily/swing ML pipeline,
analyzer UI, live-data integration) is being **replanned from scratch** — it did
not meet the complete requirement. Before the new plan is built, the workspace
must be cleaned of everything tied only to the old, abandoned designs.

---

## Steps

### 1. Delete all old plan documents and references
- Remove every superseded plan file and its references (e.g. the `PLAN_*.md`
  family, supplementary/spec docs, status snapshots) that belong to the
  abandoned designs.
- Remove references to these documents from any remaining files.

### 2. Remove other old documents, designs, and related files
- Remove any remaining documentation, design notes, diagrams, and related
  source/config/report files that are **not updated and not linked** to the new
  plan (the new plan will be created in the next step).

### 3. Reorganize & format the workspace
- Arrange all remaining documents and files into a clean, well-structured
  layout (clear folders, consistent naming, tidy root).

### 4. Delete old / unnecessary data files
- Delete old data files that are **not at all required or important** for the
  new plan (stale caches, obsolete datasets, dead artifacts, etc.).

---

## Guardrails (how this will be executed safely)

- **Nothing is deleted until the new requirement is provided** and the new plan
  defines what to keep vs. remove.
- Before deleting, a **proposed delete/keep list will be shown for approval** —
  no bulk removal without sign-off.
- **Git history is preserved**, so anything removed can be recovered if needed.

---

## Next step (owner: user)

➡️ **Provide the latest updated requirement / new plan.** Once received, this
document's steps 1–4 will be executed in order, with a keep/delete list confirmed
before anything is removed.
