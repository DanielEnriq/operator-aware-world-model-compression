from __future__ import annotations

import torch

from oawc.compression.operator_eval import compute_operator_metrics


def test_compute_operator_metrics_perfect_alignment() -> None:
    teacher = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    student = teacher.clone()
    # [N, C, H, A]
    candidates = torch.zeros((2, 4, 1, 2), dtype=torch.float32)
    candidates[:, :, 0, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    candidates[:, :, 0, 1] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    metrics = compute_operator_metrics(
        teacher_costs=teacher,
        student_costs=student,
        candidate_actions=candidates,
    )

    assert metrics["finite_student_costs"] is True
    assert metrics["raw_cost_mse"] == 0.0
    assert metrics["teacher_regret"]["mean"] == 0.0
    assert metrics["teacher_best_index_match_rate"] == 1.0
    assert metrics["spearman_per_state"]["mean"] == 1.0
    assert metrics["topk_overlap"]["1"]["mean"] == 1.0
