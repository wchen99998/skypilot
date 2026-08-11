"""Unit tests for sky.provision.slurm.instance."""
from unittest import mock

import pytest

from sky import exceptions
from sky.provision.slurm import instance
from sky.utils import status_lib

_CLUSTER = 'test-cluster'
_PROVIDER_CONFIG = {
    'cluster': _CLUSTER,
    'ssh': {
        'hostname': 'localhost',
        'port': '22',
        'user': 'testuser',
        'private_key': '/path/to/key',
    }
}


def _job_info(state):
    return instance.slurm.JobInfo('123', state, None, None)


@pytest.fixture
def mock_client(monkeypatch):
    """Mock SlurmClient for terminate_instances tests (SSH path)."""
    client = mock.MagicMock()
    monkeypatch.setattr(instance.slurm, 'SlurmClient',
                        mock.MagicMock(return_value=client))
    monkeypatch.setattr(instance.slurm_utils, 'is_inside_slurm_cluster',
                        mock.MagicMock(return_value=False))
    # Make waits resolve quickly in tests that exercise timeouts.
    monkeypatch.setattr(instance, '_TERMINATION_GRACE_PERIOD_SECONDS', 0.05)
    monkeypatch.setattr(instance, '_JOB_TERMINATION_TIMEOUT_SECONDS', 0.05)
    monkeypatch.setattr(instance, 'POLL_INTERVAL_SECONDS', 0.01)
    return client


class TestTerminateInstances:
    """Test terminate_instances() cancellation and escalation logic."""

    @pytest.mark.parametrize('job_state', [
        'COMPLETED',
        'CANCELLED',
        'FAILED',
        'TIMEOUT',
        'NODE_FAIL',
        'PREEMPTED',
        'SPECIAL_EXIT',
        'COMPLETING',
    ])
    def test_terminal_or_completing_state_no_cancel(self, mock_client,
                                                    job_state):
        mock_client.get_jobs_info_by_name.return_value = [_job_info(job_state)]
        instance.terminate_instances(_CLUSTER, provider_config=_PROVIDER_CONFIG)
        mock_client.cancel_jobs_by_name.assert_not_called()

    def test_no_jobs_found_no_cancel(self, mock_client):
        mock_client.get_jobs_info_by_name.return_value = []
        instance.terminate_instances(_CLUSTER, provider_config=_PROVIDER_CONFIG)
        mock_client.cancel_jobs_by_name.assert_not_called()

    @pytest.mark.parametrize('job_state', ['PENDING', 'CONFIGURING'])
    def test_pending_cancels_without_signal(self, mock_client, job_state):
        mock_client.get_jobs_info_by_name.return_value = [_job_info(job_state)]
        instance.terminate_instances(_CLUSTER, provider_config=_PROVIDER_CONFIG)
        mock_client.cancel_jobs_by_name.assert_called_once_with(_CLUSTER,
                                                                signal=None)

    @pytest.mark.parametrize('job_state', ['STAGING_OUT', 'SIGNALING'])
    def test_transient_state_single_graceful_cancel(self, mock_client,
                                                    job_state):
        # Transient states get the graceful signal but no verify/escalate.
        mock_client.get_jobs_info_by_name.return_value = [_job_info(job_state)]
        instance.terminate_instances(_CLUSTER, provider_config=_PROVIDER_CONFIG)
        mock_client.cancel_jobs_by_name.assert_called_once_with(_CLUSTER,
                                                                signal='TERM',
                                                                full=True)
        mock_client.get_jobs_state_by_name.assert_not_called()

    def test_autodown_inside_cluster_single_graceful_cancel(
            self, mock_client, monkeypatch):
        # Inside the cluster (autodown), the Skylet performing the teardown
        # runs inside a job step, so no step-level TERM and no
        # verify/escalate.
        monkeypatch.setattr(instance.slurm_utils, 'is_inside_slurm_cluster',
                            mock.MagicMock(return_value=True))
        mock_client.get_jobs_info_by_name.return_value = [_job_info('RUNNING')]
        instance.terminate_instances(_CLUSTER, provider_config=_PROVIDER_CONFIG)
        mock_client.cancel_jobs_by_name.assert_called_once_with(_CLUSTER,
                                                                signal='TERM',
                                                                full=True)
        mock_client.get_jobs_state_by_name.assert_not_called()

    @pytest.mark.parametrize('job_state', ['RUNNING', 'SUSPENDED'])
    def test_graceful_termination_succeeds(self, mock_client, job_state):
        # The job exits (gone from squeue) within the grace period.
        mock_client.get_jobs_info_by_name.return_value = [_job_info(job_state)]
        mock_client.get_jobs_state_by_name.return_value = []
        instance.terminate_instances(_CLUSTER, provider_config=_PROVIDER_CONFIG)
        assert mock_client.cancel_jobs_by_name.call_args_list == [
            mock.call(_CLUSTER, signal='TERM'),
            mock.call(_CLUSTER, signal='TERM', full=True),
        ]

    def test_graceful_wait_tolerates_transient_query_failure(self, mock_client):
        # One failed poll inside the wait must not fail the teardown.
        mock_client.get_jobs_info_by_name.return_value = [_job_info('RUNNING')]
        mock_client.get_jobs_state_by_name.side_effect = [
            exceptions.CommandError(255, 'squeue', 'ssh dropped', None),
            [],
        ]
        instance.terminate_instances(_CLUSTER, provider_config=_PROVIDER_CONFIG)
        assert mock_client.cancel_jobs_by_name.call_args_list == [
            mock.call(_CLUSTER, signal='TERM'),
            mock.call(_CLUSTER, signal='TERM', full=True),
        ]

    def test_escalates_when_job_survives_term(self, mock_client):
        # The job stays RUNNING through the grace period (e.g. a step
        # survived the TERM and blocks the batch script from exiting), then
        # exits after the plain scancel.
        mock_client.get_jobs_info_by_name.return_value = [_job_info('RUNNING')]

        def get_states(job_name):
            del job_name
            plain_scancel_issued = mock.call(_CLUSTER) in (
                mock_client.cancel_jobs_by_name.call_args_list)
            return [] if plain_scancel_issued else ['RUNNING']

        mock_client.get_jobs_state_by_name.side_effect = get_states
        instance.terminate_instances(_CLUSTER, provider_config=_PROVIDER_CONFIG)
        assert mock_client.cancel_jobs_by_name.call_args_list == [
            mock.call(_CLUSTER, signal='TERM'),
            mock.call(_CLUSTER, signal='TERM', full=True),
            mock.call(_CLUSTER),
        ]

    def test_raises_when_escalation_fails(self, mock_client):
        # The job keeps RUNNING even after the plain scancel.
        mock_client.get_jobs_info_by_name.return_value = [_job_info('RUNNING')]
        mock_client.get_jobs_state_by_name.return_value = ['RUNNING']
        with pytest.raises(RuntimeError, match='still.*running'):
            instance.terminate_instances(_CLUSTER,
                                         provider_config=_PROVIDER_CONFIG)
        assert mock.call(_CLUSTER) in (
            mock_client.cancel_jobs_by_name.call_args_list)

    def test_raises_when_job_wedges_in_completing(self, mock_client):
        # COMPLETING satisfies the grace-period wait (Slurm is tearing the
        # job down), but after escalation a job wedged in COMPLETING still
        # holds the allocation and must be reported, not treated as
        # success.
        mock_client.get_jobs_info_by_name.return_value = [_job_info('RUNNING')]

        def get_states(job_name):
            del job_name
            plain_scancel_issued = mock.call(_CLUSTER) in (
                mock_client.cancel_jobs_by_name.call_args_list)
            return ['COMPLETING'] if plain_scancel_issued else ['RUNNING']

        mock_client.get_jobs_state_by_name.side_effect = get_states
        with pytest.raises(RuntimeError, match='still.*running'):
            instance.terminate_instances(_CLUSTER,
                                         provider_config=_PROVIDER_CONFIG)


class TestWaitForJobStates:
    """Test _wait_for_job_states()."""

    def test_gone_from_queue(self):
        client = mock.MagicMock()
        client.get_jobs_state_by_name.return_value = []
        assert instance._wait_for_job_states(client,
                                             'job',
                                             instance._EXITING_JOB_STATES,
                                             timeout=1)

    def test_completing_counts_as_exiting(self):
        client = mock.MagicMock()
        client.get_jobs_state_by_name.return_value = ['COMPLETING']
        assert instance._wait_for_job_states(client,
                                             'job',
                                             instance._EXITING_JOB_STATES,
                                             timeout=1)

    def test_completing_is_not_terminal(self):
        client = mock.MagicMock()
        client.get_jobs_state_by_name.return_value = ['COMPLETING']
        assert not instance._wait_for_job_states(
            client, 'job', instance._TERMINAL_JOB_STATES, timeout=0)

    def test_running_times_out(self):
        client = mock.MagicMock()
        client.get_jobs_state_by_name.return_value = ['RUNNING']
        assert not instance._wait_for_job_states(
            client, 'job', instance._EXITING_JOB_STATES, timeout=0)

    def test_query_failure_until_deadline_returns_false(self):
        client = mock.MagicMock()
        client.get_jobs_state_by_name.side_effect = exceptions.CommandError(
            255, 'squeue', 'ssh dropped', None)
        assert not instance._wait_for_job_states(
            client, 'job', instance._EXITING_JOB_STATES, timeout=0)


def test_wait_for_job_nodes_pauses_on_ssh_failure():
    client = mock.MagicMock(poll_interval=60)
    client.get_job_info.side_effect = exceptions.CommandError(
        255, 'squeue', 'ssh dropped', None)

    with pytest.raises(exceptions.ExecutionPausedError,
                       match='waiting for Slurm job') as exc_info:
        instance._wait_for_job_nodes(client, '123', -1, 'workq', mock.Mock())

    assert 'without releasing' in exc_info.value.hint


def test_wait_for_job_ready_pauses_on_ssh_failure():
    runner = mock.MagicMock()
    runner.run.return_value = (1, '', '')
    client = mock.MagicMock(poll_interval=60)
    client.get_job_info.side_effect = exceptions.CommandError(
        255, 'squeue', 'ssh dropped', None)

    with pytest.raises(exceptions.ExecutionPausedError,
                       match='initializing Slurm job'):
        instance._wait_for_job_ready(runner, client, '123', '/ready',
                                     '/slurm.log')


@pytest.mark.parametrize(
    'runtime_ready,expected_status',
    [(False, status_lib.ClusterStatus.INIT),
     (True, status_lib.ClusterStatus.UP)],
)
def test_query_instances_requires_runtime_marker(monkeypatch, runtime_ready,
                                                 expected_status):
    client = mock.MagicMock()
    client.get_jobs_info_by_name.return_value = [
        instance.slurm.JobInfo('123', 'RUNNING', 'node1', None)
    ]
    client.get_remote_home_dir.return_value = '/home/testuser'
    client.check_file_exists.return_value = runtime_ready
    monkeypatch.setattr(instance.slurm, 'SlurmClient',
                        mock.MagicMock(return_value=client))
    monkeypatch.setattr(instance.skypilot_config, 'get_effective_region_config',
                        mock.MagicMock(return_value=None))
    provider_config = {
        **_PROVIDER_CONFIG,
        'poll_interval': 0,
    }

    statuses = instance.query_instances('display-name',
                                        'cluster-on-cloud',
                                        provider_config=provider_config)

    reason = None if runtime_ready else 'SkyPilot runtime setup incomplete'
    assert statuses == {
        'job123-node1': (expected_status, reason),
    }
    client.check_file_exists.assert_called_once_with(
        '/home/testuser/.sky_clusters/cluster-on-cloud/'
        '.sky/skylet_start.sh')
