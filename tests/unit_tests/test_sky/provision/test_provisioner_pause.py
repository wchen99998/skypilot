"""bulk_provision must not tear down resources when execution pauses.

A paused execution (ExecutionPausedError) is waiting on an external condition
and wants its partially provisioned resources kept so it can resume. This pins
that bulk_provision re-raises the pause without tearing down, while still
tearing down on failures and cancellation.
"""
import contextlib
from unittest import mock

import pytest

from sky import clouds
from sky import exceptions
from sky import global_user_state
from sky import resources as resources_lib
from sky.provision import provisioner
from sky.utils import resources_utils

_CLUSTER_YAML_DICT = {
    'head_node_type': 'ray.head.default',
    'provider': {},
    'auth': {},
    'docker': {},
    'available_node_types': {
        'ray.head.default': {
            'node_config': {}
        }
    },
}


@pytest.fixture()
def patched_bulk_provision(monkeypatch):
    """Drive bulk_provision with its filesystem/state deps stubbed out.

    Returns the teardown_cluster mock so tests can assert on it; the caller
    sets _bulk_provision's side effect.
    """
    monkeypatch.setattr(global_user_state, 'get_cluster_yaml_dict',
                        lambda *a, **k: dict(_CLUSTER_YAML_DICT))
    monkeypatch.setattr(provisioner.provision_logging,
                        'setup_provision_logging',
                        lambda *a, **k: contextlib.nullcontext())
    teardown_mock = mock.MagicMock()
    monkeypatch.setattr(provisioner, 'teardown_cluster', teardown_mock)
    return teardown_mock


def _call_bulk_provision(tmp_path, resources=None, prev_cluster_ever_up=False):
    cloud = clouds.Kubernetes()
    if resources is None:
        resources = resources_lib.Resources(cloud=cloud, instance_type='dummy')
    else:
        assert resources.cloud is not None
        cloud = resources.cloud
    return provisioner.bulk_provision(cloud=cloud,
                                      region=clouds.Region('us'),
                                      zones=None,
                                      cluster_name=resources_utils.ClusterName(
                                          'c', 'c-on-cloud'),
                                      resources=resources,
                                      num_nodes=1,
                                      cluster_yaml='/fake/cluster.yaml',
                                      prev_cluster_ever_up=prev_cluster_ever_up,
                                      log_dir=str(tmp_path))


def test_bulk_provision_does_not_teardown_on_pause(patched_bulk_provision,
                                                   monkeypatch, tmp_path):
    """A pause propagates without tearing down the kept resources."""
    paused = exceptions.ExecutionPausedError('Waiting on admission.',
                                             hint='resume later',
                                             retry_wait_seconds=5)
    monkeypatch.setattr(provisioner, '_bulk_provision',
                        mock.MagicMock(side_effect=paused))

    with pytest.raises(exceptions.ExecutionPausedError):
        _call_bulk_provision(tmp_path)

    patched_bulk_provision.assert_not_called()


def test_bulk_provision_tears_down_on_ordinary_failure(patched_bulk_provision,
                                                       monkeypatch, tmp_path):
    """Negative control: an ordinary failure still tears down.

    Proves the test harness actually reaches the teardown branch, so the
    pause test above is meaningful rather than superfluous.
    """
    monkeypatch.setattr(
        provisioner, '_bulk_provision',
        mock.MagicMock(side_effect=RuntimeError('provisioning failed')))

    with pytest.raises(RuntimeError, match='provisioning failed'):
        _call_bulk_provision(tmp_path)

    patched_bulk_provision.assert_called_once()


def test_bulk_provision_terminates_previously_up_compute_tpu_after_failure(
        patched_bulk_provision, monkeypatch, tmp_path):
    """A failed Compute TPU must be deleted before it can be recreated."""
    monkeypatch.setattr(
        provisioner, '_bulk_provision',
        mock.MagicMock(side_effect=RuntimeError('provisioning failed')))
    resources = resources_lib.Resources(cloud=clouds.GCP(),
                                        instance_type='ct6e-standard-4t')

    with pytest.raises(RuntimeError, match='provisioning failed'):
        _call_bulk_provision(tmp_path,
                             resources=resources,
                             prev_cluster_ever_up=True)

    patched_bulk_provision.assert_called_once_with('GCP',
                                                   resources_utils.ClusterName(
                                                       'c', 'c-on-cloud'),
                                                   terminate=True,
                                                   provider_config={})


def test_bulk_provision_stops_other_previously_up_resources_after_failure(
        patched_bulk_provision, monkeypatch, tmp_path):
    """Previously-UP resources keep the normal state-preserving stop path."""
    monkeypatch.setattr(
        provisioner, '_bulk_provision',
        mock.MagicMock(side_effect=RuntimeError('provisioning failed')))
    resources = resources_lib.Resources(cloud=clouds.Kubernetes(),
                                        instance_type='dummy')

    with pytest.raises(RuntimeError, match='provisioning failed'):
        _call_bulk_provision(tmp_path,
                             resources=resources,
                             prev_cluster_ever_up=True)

    patched_bulk_provision.assert_called_once_with('Kubernetes',
                                                   resources_utils.ClusterName(
                                                       'c', 'c-on-cloud'),
                                                   terminate=False,
                                                   provider_config={})


def test_bulk_provision_tears_down_on_keyboard_interrupt(
        patched_bulk_provision, monkeypatch, tmp_path):
    """API-request cancellation must roll back accepted cloud resources."""
    monkeypatch.setattr(
        provisioner, '_bulk_provision',
        mock.MagicMock(side_effect=KeyboardInterrupt('request cancelled')))

    with pytest.raises(KeyboardInterrupt, match='request cancelled'):
        _call_bulk_provision(tmp_path)

    patched_bulk_provision.assert_called_once()


def test_bulk_provision_leaves_teardown_to_down_on_cancellation(
        patched_bulk_provision, monkeypatch, tmp_path):
    """A concurrent sky down is the sole teardown owner."""
    monkeypatch.setattr(
        provisioner, '_bulk_provision',
        mock.MagicMock(side_effect=KeyboardInterrupt('request cancelled')))
    monkeypatch.setattr(provisioner, '_cluster_teardown_owns_cancellation',
                        lambda: True)

    with pytest.raises(KeyboardInterrupt, match='request cancelled'):
        _call_bulk_provision(tmp_path)

    patched_bulk_provision.assert_not_called()
