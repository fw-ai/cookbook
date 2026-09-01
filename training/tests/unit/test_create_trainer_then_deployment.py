from types import SimpleNamespace

from training.examples.tools.create_trainer_then_deployment import (
    create_trainer_then_deployment,
)


def test_waits_for_trainer_before_creating_deployment() -> None:
    events: list[str] = []
    profile = SimpleNamespace(
        training_shape_version="accounts/fireworks/trainingShapes/shape/versions/v1",
        deployment_shape="accounts/fireworks/deploymentShapes/shape/versions/v1",
    )
    trainer_endpoint = SimpleNamespace(
        job_id="trainer-1",
        job_name="accounts/account/rlorTrainerJobs/trainer-1",
        base_url="https://trainer.example.com",
    )
    ready_deployment = SimpleNamespace(
        name="accounts/account/deployments/deployment-1",
        inference_model="accounts/account/deployments/deployment-1",
    )

    class FakeTrainerManager:
        def resolve_training_profile(self, training_shape):
            events.append("resolve-shape")
            assert training_shape == "accounts/fireworks/trainingShapes/shape"
            return profile

        def create(self, config):
            events.append("create-trainer")
            assert config.use_reservation is False
            return SimpleNamespace(
                job_id=trainer_endpoint.job_id,
                job_name=trainer_endpoint.job_name,
            )

        def wait_for_ready(self, job_id, **kwargs):
            events.append("wait-trainer")
            assert job_id == trainer_endpoint.job_id
            assert kwargs["job_name"] == trainer_endpoint.job_name
            return trainer_endpoint

    class FakeDeploymentManager:
        def create_or_get(self, config):
            events.append("create-deployment")
            assert config.hot_load_trainer_job == trainer_endpoint.job_name
            assert config.extra_values == {"bypass_reservation": "true"}
            return SimpleNamespace(state="CREATING")

        def wait_for_ready(self, deployment_id, **_kwargs):
            events.append("wait-deployment")
            assert deployment_id == "deployment-1"
            return ready_deployment

    trainer, deployment = create_trainer_then_deployment(
        trainer_manager=FakeTrainerManager(),
        deployment_manager=FakeDeploymentManager(),
        base_model="accounts/fireworks/models/model",
        training_shape="accounts/fireworks/trainingShapes/shape",
        deployment_id="deployment-1",
        bypass_reservation=True,
    )

    assert trainer is trainer_endpoint
    assert deployment is ready_deployment
    assert events == [
        "resolve-shape",
        "create-trainer",
        "wait-trainer",
        "create-deployment",
        "wait-deployment",
    ]
