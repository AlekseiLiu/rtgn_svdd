import pytest
import torch

from rtgn_svdd.objectives import (
    LOG_VARIANCE_MAX,
    LOG_VARIANCE_MIN,
    deterministic_svdd_loss,
    robust_training_loss,
    initialize_center,
    negative_uncertainty_nll,
    positive_gaussian_nll,
    sample_negative_targets,
    variance_from_log_variance,
)


def test_positive_gaussian_nll_matches_variance_formula_and_has_gradients():
    event_mean = torch.tensor([[1.0, 3.0]], requires_grad=True)
    center = torch.tensor([0.0, 1.0], requires_grad=True)
    log_variance = torch.log(torch.tensor([[1.0, 4.0]])).requires_grad_()

    loss = positive_gaussian_nll(event_mean, center, log_variance)

    expected = 0.5 * torch.tensor([1.0, torch.log(torch.tensor(4.0)) + 1.0]).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert event_mean.grad is not None
    assert center.grad is not None
    assert log_variance.grad is not None


def test_negative_nll_uses_sampled_target_and_only_trains_uncertainty():
    target = torch.tensor([[2.0, 4.0]])
    log_variance = torch.log(torch.tensor([[4.0, 16.0]])).requires_grad_()

    loss = negative_uncertainty_nll(target, log_variance)

    expected = 0.5 * (log_variance.detach() + 1.0).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert log_variance.grad is not None


def test_sampling_primitives_are_generator_driven_and_use_covariance_five():
    center_a = initialize_center(4, generator=torch.Generator().manual_seed(7))
    center_b = initialize_center(4, generator=torch.Generator().manual_seed(7))
    target_a = sample_negative_targets(3, 4, generator=torch.Generator().manual_seed(9))
    base = torch.randn((3, 4), generator=torch.Generator().manual_seed(9))

    assert torch.equal(center_a, center_b)
    assert torch.allclose(target_a, base * torch.sqrt(torch.tensor(5.0)))


def test_negative_targets_can_use_an_output_device_distinct_from_generator():
    target = sample_negative_targets(
        3,
        4,
        generator=torch.Generator(device="cpu").manual_seed(9),
        device="meta",
    )

    assert target.shape == (3, 4)
    assert target.device.type == "meta"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@torch.no_grad()
def test_cpu_generator_can_drive_cuda_negative_targets_when_available():
    target = sample_negative_targets(
        3,
        4,
        generator=torch.Generator(device="cpu").manual_seed(9),
        device="cuda",
    )
    expected = sample_negative_targets(
        3,
        4,
        generator=torch.Generator(device="cpu").manual_seed(9),
        device="cpu",
    )

    assert target.device.type == "cuda"
    assert torch.equal(target.cpu(), expected)


def test_robust_training_loss_adds_separately_averaged_branches():
    positive_mean = torch.tensor([[1.0, 1.0]])
    center = torch.zeros(2)
    positive_log_variance = torch.zeros_like(positive_mean)
    negative_target = torch.tensor([[2.0, 2.0], [4.0, 4.0]])
    negative_log_variance = torch.zeros_like(negative_target)

    loss = robust_training_loss(
        positive_mean,
        center,
        positive_log_variance,
        negative_target,
        negative_log_variance,
    )

    assert torch.allclose(loss, torch.tensor(5.5))  # 0.5 + mean([2, 8])


def test_variance_conversion_clamps_extreme_values_in_one_place():
    converted = variance_from_log_variance(torch.tensor([-100.0, 100.0]))
    expected = torch.exp(torch.tensor([LOG_VARIANCE_MIN, LOG_VARIANCE_MAX]))
    assert torch.equal(converted, expected)
    assert torch.isfinite(converted).all()

    finite_loss = positive_gaussian_nll(
        torch.zeros((1, 2)), torch.zeros(2), torch.tensor([[-100.0, 100.0]])
    )
    assert torch.isfinite(finite_loss)


def test_deterministic_loss_is_mean_squared_distance_not_fourth_power():
    scores = torch.tensor([1.0, 9.0], requires_grad=True)
    loss = deterministic_svdd_loss(scores)
    assert torch.equal(loss, torch.tensor(5.0))
    loss.backward()
    assert torch.equal(scores.grad, torch.tensor([0.5, 0.5]))
