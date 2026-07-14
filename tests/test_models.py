import torch

from rtgn_svdd.models import RobustSVDDHead, SVDDHead, TemporalGraphEncoder


def test_deterministic_head_concatenates_endpoints_and_scores_center_distance():
    center = torch.tensor([1.0, 1.0, 2.0, 2.0])
    head = SVDDHead(endpoint_dim=2, center=center)
    nodes = torch.tensor([[1.0, 3.0], [4.0, 2.0], [0.0, 1.0]])

    output = head(nodes, src=torch.tensor([0, 2]), dst=torch.tensor([1, 0]))

    expected_events = torch.tensor([[1.0, 3.0, 4.0, 2.0], [0.0, 1.0, 1.0, 3.0]])
    assert torch.equal(output.event_mean, expected_events)
    assert torch.equal(output.scores.s_mu, torch.tensor([8.0, 3.0]))
    assert output.scores.s_sigma is None


def test_robust_head_uses_per_dimension_log_variance_by_default():
    head = RobustSVDDHead(endpoint_dim=2, center=torch.zeros(4))
    # Each row contains two means followed by two log-variances.
    nodes = torch.tensor([[1.0, 2.0, 0.0, 0.6931472], [3.0, 4.0, 1.0986123, 1.3862944]])

    output = head(nodes, src=torch.tensor([0]), dst=torch.tensor([1]))

    assert torch.allclose(output.event_mean, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert torch.allclose(
        output.event_log_variance,
        torch.tensor([[0.0, 0.6931472, 1.0986123, 1.3862944]]),
    )
    assert torch.allclose(output.scores.s_mu, torch.tensor([30.0]))
    assert torch.allclose(output.scores.s_sigma, torch.tensor([2.5]))


def test_robust_head_preserves_historical_shared_per_node_mode():
    head = RobustSVDDHead(
        endpoint_dim=2,
        center=torch.zeros(4),
        uncertainty_mode="historical_shared_per_node",
    )
    nodes = torch.tensor([[1.0, 2.0, 0.0, 2.0], [3.0, 4.0, 2.0, 4.0]])

    output = head(nodes, src=torch.tensor([0]), dst=torch.tensor([1]))

    assert torch.equal(output.event_log_variance, torch.tensor([[1.0, 1.0, 3.0, 3.0]]))


def test_shared_temporal_encoder_builds_relative_time_edge_features():
    class TimeEncoder(torch.nn.Module):
        def forward(self, relative_time):
            return relative_time.unsqueeze(-1)

    class Convolution(torch.nn.Module):
        def forward(self, node_memory, edge_index, edge_features):
            assert torch.equal(edge_features, torch.tensor([[7.0, 4.0], [16.0, 5.0]]))
            return node_memory + 1.0

    encoder = TemporalGraphEncoder(TimeEncoder(), Convolution())
    node_memory = torch.tensor([[1.0], [2.0], [3.0]])
    last_update = torch.tensor([10.0, 20.0, 30.0])
    edge_index = torch.tensor([[0, 1], [1, 2]])

    output = encoder(
        node_memory,
        last_update,
        edge_index,
        event_time=torch.tensor([3.0, 4.0]),
        message=torch.tensor([[4.0], [5.0]]),
    )

    assert torch.equal(output, node_memory + 1.0)
