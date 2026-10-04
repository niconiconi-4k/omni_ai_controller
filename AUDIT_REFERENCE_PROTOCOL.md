# Audit short references v1 / agent state v2

Main integration contract (Controller-only implementation; no DB/API key change).

## Complete state schema

`agent_state` is a JSON object:

- `version`: integer `2` (readers accept old `1` for migration only).
- `reference_schema`: string `audit-short-references-v1`.
- `audit_id`, `run_id`: display references. Resolve through `identity_map.reverse`
  before checking the real request scope. If audit and fallback run are the same
  real identity, they deliberately share one reference.
- `identity_map`: **internal transport metadata, never model/book/UI prose**:
  - `version`: integer `1`.
  - `forward`: object, exact original string → display reference.
  - `reverse`: object, display reference → exact original string.
  - `next`: object, single-letter namespace → next positive integer.
    Counters are monotonically increasing and repaired upward from `reverse`.
  - forward/reverse must be bijective; corruption fails closed (no renumbering).
- `notebooks`: object with `audit_planner` and `evidence_worker`, each:
  - `limit_bytes`: integer `8388608`.
  - `used_bytes`: complete compact sorted JSON UTF-8 byte count, including envelope.
  - `evicted_entries`, `overflow`: nonnegative integers.
  - `entries`: array of `{key, kind, content, source_fingerprint}`.
    `key` is B or Q reference; `source_fingerprint` is F reference or empty string;
    content recursively uses references including dynamic UUID object keys/prose.
   Non-ID financial text uses reversible literal escaping in storage only:
   `~` → `~~`, an original literal single-letter numeric token → `~token`.
   This distinguishes an actual accounting reference `R001` from the display
   label of a UUID. Decode consumes escaped segments atomically before restoring
   unescaped labels. Decode v2 only; v1 has no escapes. Model projections are
   built from decoded facts and use **no literal escaping**.
- `task_lists`: object with both owners → array of tasks (maximum 512 per owner).
  Existing task fields remain unchanged in shape; identifiers, dependencies,
  parent/derived IDs, source scope and source_fingerprint recursively use references.
- `stats`: existing integer counters `cache_hits`, `cache_misses`, `invalidations`,
  `amount_searches`, `exact_amount_hits`, `duplicate_candidates`, `task_evictions`,
  `task_overflow`, `evidence_reuses`.

## Namespace and stability

T transaction, R receipt/child (including `upload:child_index` as one identity),
G candidate/match group, U upload, S source/other ID/free-text UUID,
A audit, N run, K task/dependency, B book entry/source cache key,
Q search cache key, F full SHA-256 fingerprint.
Use one uppercase letter followed by decimal digits; initial padding is 3 digits
(`T001`), IDs beyond 999 grow normally. Existing map aliases must be retained.
The same exact original identity always has the same reference even if another
field requests a different namespace. These references are labels, not financial
evidence and not global DB IDs.

Seed T/R from **complete source_inventory in inventory order including confirmed
sources**, then residual rows and candidates. Without inventory sort original
source IDs deterministically solely to allocate display labels. This sorting must
not break financial ambiguity or alter confidence. Keep identity_map across
iterations/residual filtering and same-audit new runs; append only. New runs clear
book/task caches (case/run-bound) but retain same-audit aliases. Never reuse maps
from another case.

## Main compatibility requirements

1. Persist/return the entire v2 state and identity_map without exposing the map to
   models or rendering raw values in notebooks/tasklists. Allow v2 in state validators.
2. Case/run validation uses reverse mapping, **not** string comparison of short
   state scope with request UUID. The outer request and authoritative approval
   steps/results still use exact real DB/API IDs. Approval gates are unchanged.
3. Mirrored `register_inventory` must implement this same storage adapter:
   decode trusted stored books/tasks, register unchanged canonical source facts,
   hash complete original source with existing canonical_bytes/fingerprint,
   encode derivatives, then recompute byte size. Do not hash display projections.
4. Fingerprint comparison resolves F → complete 64-hex digest; compare the full
   value. Store neither digest nor raw cache key in entries/tasklists. Keep map
   identities even if individual derivative entries are evicted.
5. v1 migration preserves **all** existing evidence/task content and complete
   hashes, shortens recursively, preserves counters and remeasures books. Cached
   unchanged source facts must hit; changed facts invalidate basic AND details.
6. Internal facts/prepare/scheduler projections restore real scope IDs; only the
   display/model projection is short. Unknown model reference must not register
   itself as an approved source; reject it via unchanged real candidate scope.
   Group IDs are reverse-resolved before atomic channel-group validation.
   Read-modify-write task transitions must decode stored tasks first, otherwise
   literal escapes would accumulate. `skill_key` uses the same S mapping as
   `approved_skill_ids`; do not present inconsistent labels for the same skill.
7. No UUID in financial disambiguation, no inference from reference order, no
   confidence changes; monetary/date strings are not replacement targets.

No approval decisions are moved into evictable caches. identity_map is sensitive
internal metadata and needs existing internal-only API authorization. Its growth
is not bounded by the per-book 8 MiB limit; future compaction must not renumber or
discard identities still referenced by retained state or authoritative steps.