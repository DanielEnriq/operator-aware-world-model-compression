# SWM PreJEPA Checkpoint Audit Plan

This stage is audit-only. We are not running compression yet.

## Why audit first

LeWM checkpoints already fit the current closed-loop cost-model flow.
SWM PreJEPA / DINO-WM checkpoints may use different object formats and may
not expose a direct `get_cost(info_dict, action_candidates)` interface.

Before compression wiring, we need to identify:

- which artifact is the best load target (`*_object.ckpt`, `weights_epoch_*.pt`,
  or `*_weights.ckpt`)
- whether it loads as a full callable `torch.nn.Module` or a state-dict-style
  checkpoint
- which interface mode it exposes (`cost_model_direct`, `forward_only`,
  `representation_rollout_only`, or `no_planning_interface`)
- which module groups are credible first compression targets

## Current audit outputs

`scripts/audit_swm_prejepa_checkpoint.py` writes:

- `outputs/audits/swm_prejepa/tworoom_checkpoint_audit.json`
- `outputs/audits/swm_prejepa/tworoom_checkpoint_audit.md`
- `outputs/audits/swm_prejepa/tworoom_compression_targets.json`

These outputs are designed to answer:

1. Is the checkpoint loadable as a full model object?
2. If loadable, can it support closed-loop planning directly?
3. If not, should we evaluate prediction/rollout fidelity first?
4. Which submodules should be compressed first?

## Gating for next step

Compression runner work starts only after the audit confirms a stable load path
and a clear interface mode. If no direct planning-cost interface exists, the
next step should focus on prediction/representation fidelity evaluation before
planner-facing compression claims.
