from experiments.learning.run_model_learning_experiments import (
    build_experiment_specs,
    filter_experiment_specs,
    run_experiment_spec,
)


def test_build_experiment_specs_includes_all_model_families():
    specs = build_experiment_specs(quick=True)
    model_names = {spec.model_name for spec in specs}

    assert model_names == {"mlp", "cnn", "rnn", "transformer", "mini_gpt"}
    assert all(spec.task_name for spec in specs)
    assert all(spec.params for spec in specs)


def test_filter_experiment_specs_returns_requested_model_only():
    specs = build_experiment_specs(quick=True)
    filtered_specs = filter_experiment_specs(specs, model_name="mlp")

    assert filtered_specs
    assert {spec.model_name for spec in filtered_specs} == {"mlp"}


def test_run_experiment_spec_returns_learning_metrics():
    spec = filter_experiment_specs(build_experiment_specs(quick=True), model_name="mlp")[0]

    result = run_experiment_spec(spec, log_every=0)

    assert result.model_name == "mlp"
    assert result.task_name == spec.task_name
    assert result.params == spec.params
    assert result.initial_loss > result.final_loss
    assert result.loss_ratio < 0.8
    assert result.final_accuracy >= 0.9
    assert result.elapsed_seconds >= 0.0
