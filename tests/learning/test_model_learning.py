import torch
import torch.nn as nn
import torch.nn.functional as F

from modular_rl.networks import (
    make_cnn_mlp,
    make_mini_gpt,
    make_mlp,
    make_rnn,
    make_transformer,
)


def _train_classifier(model, inputs, targets, steps=120, learning_rate=0.03):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for _ in range(steps):
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def _train_sequence_classifier(model, inputs, targets, steps=160, learning_rate=0.02):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for _ in range(steps):
        logits = model(inputs)
        last_step_logits = logits[:, -1, :]
        loss = F.cross_entropy(last_step_logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def test_mlp_learns_xor_classification():
    torch.manual_seed(0)
    inputs = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    targets = torch.tensor([0, 1, 1, 0])
    model = make_mlp(input_dim=2, output_dim=2, hidden_dims=[16, 16])

    losses = _train_classifier(model, inputs, targets, steps=200, learning_rate=0.05)
    predictions = model(inputs).argmax(dim=-1)

    assert losses[-1] < losses[0] * 0.2
    assert torch.equal(predictions, targets)


def test_cnn_mlp_learns_image_corner_classification():
    torch.manual_seed(1)
    images = torch.zeros(4, 1, 8, 8)
    images[0, 0, 1, 1] = 1.0
    images[1, 0, 1, 6] = 1.0
    images[2, 0, 6, 1] = 1.0
    images[3, 0, 6, 6] = 1.0
    targets = torch.tensor([0, 1, 2, 3])
    model = make_cnn_mlp(
        input_shape=(1, 8, 8),
        output_dim=4,
        conv_channels=[8, 8],
        mlp_hidden_dims=[16],
        cnn_feature_dim=16,
    )

    losses = _train_classifier(model, images, targets, steps=160, learning_rate=0.03)
    predictions = model(images).argmax(dim=-1)

    assert losses[-1] < losses[0] * 0.25
    assert torch.equal(predictions, targets)


def test_rnn_learns_order_dependent_sequence_classification():
    torch.manual_seed(2)
    inputs = torch.tensor(
        [
            [[1.0], [0.0], [0.0]],
            [[0.0], [1.0], [0.0]],
            [[0.0], [0.0], [1.0]],
            [[1.0], [1.0], [0.0]],
            [[1.0], [0.0], [1.0]],
            [[0.0], [1.0], [1.0]],
        ]
    )
    targets = torch.tensor([0, 1, 1, 1, 0, 1])
    model = make_rnn(input_dim=1, output_dim=2, hidden_dims=[8], rnn_hidden_dim=8)

    losses = _train_sequence_classifier(model, inputs, targets, steps=180, learning_rate=0.03)
    predictions = model(inputs)[:, -1, :].argmax(dim=-1)

    assert losses[-1] < losses[0] * 0.35
    assert torch.equal(predictions, targets)


def test_transformer_learns_sequence_position_classification():
    torch.manual_seed(3)
    inputs = torch.tensor(
        [
            [[1.0], [0.0], [0.0], [0.0]],
            [[0.0], [1.0], [0.0], [0.0]],
            [[0.0], [0.0], [1.0], [0.0]],
            [[0.0], [0.0], [0.0], [1.0]],
        ]
    )
    targets = torch.tensor([0, 1, 2, 3])
    model = make_transformer(
        input_dim=1,
        output_dim=4,
        hidden_dims=[8],
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        max_seq_len=4,
        dropout=0.0,
    )

    losses = _train_sequence_classifier(model, inputs, targets, steps=180, learning_rate=0.02)
    predictions = model(inputs)[:, -1, :].argmax(dim=-1)

    assert losses[-1] < losses[0] * 0.35
    assert torch.equal(predictions, targets)


def test_mini_gpt_learns_repeating_token_pattern():
    torch.manual_seed(4)
    input_ids = torch.tensor(
        [
            [0, 1, 2, 0, 1],
            [1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0],
            [0, 1, 2, 0, 1],
        ],
        dtype=torch.long,
    )
    next_token_targets = torch.tensor(
        [
            [1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0],
            [0, 1, 2, 0, 1],
            [1, 2, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    model = make_mini_gpt(
        vocab_size=3,
        max_seq_len=5,
        embed_dim=12,
        num_heads=3,
        num_layers=1,
        dim_feedforward=24,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    losses = []

    for _ in range(180):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, 3), next_token_targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    predictions = model(input_ids).argmax(dim=-1)

    assert losses[-1] < losses[0] * 0.2
    assert torch.equal(predictions, next_token_targets)
