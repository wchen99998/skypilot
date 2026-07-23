"""Tests for GCP Flex-start MIG credential and cancellation handling."""

from unittest import mock

from google.auth import exceptions as google_auth_exceptions
import pytest

from sky.provision.gcp import constants
from sky.provision.gcp import instance_utils
from sky.provision.gcp import mig_utils


class _RefreshError(Exception):
    """Test double for google.auth.exceptions.RefreshError."""


class _DefaultCredentialsError(Exception):
    """Test double for google.auth.exceptions.DefaultCredentialsError."""


class _TransientHttpError(Exception):
    """Test double for googleapiclient.errors.HttpError."""

    def __init__(self, status):
        super().__init__(f'HTTP {status}')
        self.resp = mock.Mock(status=status)


class _TransportError(Exception):
    """Test double for httplib2.HttpLib2Error."""


class _GoogleAuthTransportError(Exception):
    """Test double for google.auth.exceptions.TransportError."""


def test_google_auth_transport_error_adaptor():
    assert (mig_utils.gcp.auth_transport_error_exception() is
            google_auth_exceptions.TransportError)


def _stable_mig_status():
    return {
        'targetSize': 1,
        'status': {
            'isStable': True,
            'bulkInstanceOperation': {
                'inProgress': False,
            },
            'currentInstanceStatuses': {
                'running': 1,
            },
        },
    }


def test_wait_reloads_adc_after_refresh_error(monkeypatch):
    stale_compute = mock.MagicMock()
    stale_execute = (stale_compute.regionInstanceGroupManagers.return_value.get.
                     return_value.execute)
    stale_execute.side_effect = _RefreshError('expired user session')

    refreshed_compute = mock.MagicMock()
    refreshed_execute = (refreshed_compute.regionInstanceGroupManagers.
                         return_value.get.return_value.execute)
    refreshed_execute.return_value = _stable_mig_status()

    build = mock.MagicMock(side_effect=[stale_compute, refreshed_compute])
    sleep = mock.MagicMock()
    monkeypatch.setattr(mig_utils.gcp, 'build', build)
    monkeypatch.setattr(mig_utils.gcp, 'gcp_auth_refresh_error_exception',
                        lambda: _RefreshError)
    monkeypatch.setattr(mig_utils.gcp, 'credential_error_exception',
                        lambda: _DefaultCredentialsError)
    monkeypatch.setattr(mig_utils.time, 'sleep', sleep)

    mig_utils.wait_for_region_managed_group_to_be_stable('project',
                                                         'us-east5',
                                                         'sky-mig-cluster',
                                                         timeout=3600)

    assert build.call_count == 2
    assert stale_execute.call_count == 1
    assert refreshed_execute.call_count == 1
    sleep.assert_called_once_with(
        mig_utils._GCP_AUTH_REFRESH_RETRY_INTERVAL_SECONDS)


def test_wait_reloads_adc_while_credential_file_is_temporarily_missing(
        monkeypatch):
    refreshed_compute = mock.MagicMock()
    refreshed_execute = (refreshed_compute.regionInstanceGroupManagers.
                         return_value.get.return_value.execute)
    refreshed_execute.return_value = _stable_mig_status()

    build = mock.MagicMock(side_effect=[
        _DefaultCredentialsError('ADC file is being replaced'),
        refreshed_compute,
    ])
    monkeypatch.setattr(mig_utils.gcp, 'build', build)
    monkeypatch.setattr(mig_utils.gcp, 'gcp_auth_refresh_error_exception',
                        lambda: _RefreshError)
    monkeypatch.setattr(mig_utils.gcp, 'credential_error_exception',
                        lambda: _DefaultCredentialsError)
    monkeypatch.setattr(mig_utils.time, 'sleep', mock.MagicMock())

    mig_utils.wait_for_region_managed_group_to_be_stable('project',
                                                         'us-east5',
                                                         'sky-mig-cluster',
                                                         timeout=3600)

    assert build.call_count == 2
    assert refreshed_execute.call_count == 1


def test_wait_reloads_adc_until_provision_deadline(monkeypatch):
    compute = mock.MagicMock()
    execute = (compute.regionInstanceGroupManagers.return_value.get.
               return_value.execute)
    execute.side_effect = _RefreshError('expired user session')
    build = mock.MagicMock(return_value=compute)
    clock = [0.0]
    sleep_calls = []

    def sleep(seconds):
        sleep_calls.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(mig_utils.gcp, 'build', build)
    monkeypatch.setattr(mig_utils.gcp, 'gcp_auth_refresh_error_exception',
                        lambda: _RefreshError)
    monkeypatch.setattr(mig_utils.gcp, 'credential_error_exception',
                        lambda: _DefaultCredentialsError)
    monkeypatch.setattr(mig_utils.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(mig_utils.time, 'sleep', sleep)
    warning = mock.MagicMock()
    monkeypatch.setattr(mig_utils.logger, 'warning', warning)

    with pytest.raises(TimeoutError, match='while reloading GCP'):
        mig_utils.wait_for_region_managed_group_to_be_stable('project',
                                                             'us-east5',
                                                             'sky-mig-cluster',
                                                             timeout=25)

    assert build.call_count == 3
    assert execute.call_count == 3
    assert sleep_calls == [10, 10, 5]
    assert warning.call_count == 1


@pytest.mark.parametrize(
    'retryable_error',
    [
        _TransientHttpError(503),
        _TransportError('connection reset'),
        _GoogleAuthTransportError('credential transport reset'),
    ],
)
def test_wait_retries_transient_api_errors_with_outer_deadline(
        monkeypatch, retryable_error):
    stale_compute = mock.MagicMock()
    stale_execute = (stale_compute.regionInstanceGroupManagers.return_value.get.
                     return_value.execute)
    stale_execute.side_effect = retryable_error
    refreshed_compute = mock.MagicMock()
    refreshed_execute = (refreshed_compute.regionInstanceGroupManagers.
                         return_value.get.return_value.execute)
    refreshed_execute.return_value = _stable_mig_status()

    build = mock.MagicMock(side_effect=[stale_compute, refreshed_compute])
    sleep = mock.MagicMock()
    monkeypatch.setattr(mig_utils.gcp, 'build', build)
    monkeypatch.setattr(mig_utils.gcp, 'gcp_auth_refresh_error_exception',
                        lambda: _RefreshError)
    monkeypatch.setattr(mig_utils.gcp, 'credential_error_exception',
                        lambda: _DefaultCredentialsError)
    monkeypatch.setattr(mig_utils.gcp, 'http_error_exception',
                        lambda: _TransientHttpError)
    monkeypatch.setattr(mig_utils.gcp, 'http_transport_error_exception',
                        lambda: _TransportError)
    monkeypatch.setattr(mig_utils.gcp, 'auth_transport_error_exception',
                        lambda: _GoogleAuthTransportError)
    monkeypatch.setattr(mig_utils.time, 'sleep', sleep)

    mig_utils.wait_for_region_managed_group_to_be_stable('project',
                                                         'us-east5',
                                                         'sky-mig-cluster',
                                                         timeout=3600)

    assert build.call_count == 2
    sleep.assert_called_once_with(
        mig_utils._GCP_TRANSIENT_RETRY_INTERVAL_SECONDS)


def test_keyboard_interrupt_is_left_to_bulk_provision_cleanup(monkeypatch):
    monkeypatch.setattr(mig_utils, 'get_missing_tpu_flex_start_permissions',
                        lambda project_id: [])
    monkeypatch.setattr(mig_utils, 'get_region_instance_template',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(mig_utils, 'get_region_managed_instance_group',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(mig_utils, 'get_workload_policy',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        lambda *args, **kwargs: [])
    monkeypatch.setattr(mig_utils, 'create_region_instance_template',
                        lambda *args, **kwargs: {'name': 'template-op'})
    monkeypatch.setattr(mig_utils, 'create_workload_policy',
                        lambda *args, **kwargs: {'name': 'policy-op'})
    monkeypatch.setattr(mig_utils, 'create_region_managed_instance_group',
                        lambda *args, **kwargs: {'name': 'mig-op'})
    monkeypatch.setattr(mig_utils, 'wait_for_region_managed_group_to_be_stable',
                        mock.MagicMock(side_effect=KeyboardInterrupt))
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'wait_for_operation',
                        classmethod(lambda cls, *args, **kwargs: None))
    cleanup = mock.MagicMock()
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, '_delete_tpu_mig_resources',
        classmethod(lambda cls, *args, **kwargs: cleanup(*args, **kwargs)))

    with pytest.raises(KeyboardInterrupt):
        instance_utils.GCPManagedInstanceGroup.create_instances(
            cluster_name='cluster',
            project_id='project',
            zone='us-east5-a',
            node_config={
                constants.MANAGED_INSTANCE_GROUP_CONFIG: {
                    'run_duration': 3600,
                    'provision_timeout': 3600,
                    'accelerator_topology': '4x4',
                },
                'machineType': 'ct6e-standard-4t',
                'labels': {},
            },
            labels={},
            count=4,
            total_count=4,
            include_head_node=True,
        )

    cleanup.assert_not_called()


def test_keyboard_interrupt_during_template_wait_is_left_to_bulk_cleanup(
        monkeypatch):
    template_operation = {'name': 'template-op'}
    monkeypatch.setattr(mig_utils, 'get_missing_tpu_flex_start_permissions',
                        lambda project_id: [])
    monkeypatch.setattr(mig_utils, 'get_region_instance_template',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(mig_utils, 'get_region_managed_instance_group',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(mig_utils, 'get_workload_policy',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        lambda *args, **kwargs: [])
    monkeypatch.setattr(mig_utils, 'create_region_instance_template',
                        lambda *args, **kwargs: template_operation)
    wait_for_operation = mock.MagicMock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, 'wait_for_operation',
        classmethod(
            lambda cls, *args, **kwargs: wait_for_operation(*args, **kwargs)))
    cleanup = mock.MagicMock()
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, '_delete_tpu_mig_resources',
        classmethod(lambda cls, *args, **kwargs: cleanup(*args, **kwargs)))

    with pytest.raises(KeyboardInterrupt):
        instance_utils.GCPManagedInstanceGroup.create_instances(
            cluster_name='cluster',
            project_id='project',
            zone='us-east5-a',
            node_config={
                constants.MANAGED_INSTANCE_GROUP_CONFIG: {
                    'run_duration': 3600,
                    'provision_timeout': 3600,
                    'accelerator_topology': '4x4',
                },
                'machineType': 'ct6e-standard-4t',
                'labels': {},
            },
            labels={},
            count=4,
            total_count=4,
            include_head_node=True,
        )

    wait_for_operation.assert_called_once_with(template_operation,
                                               'project',
                                               region='us-east5')
    cleanup.assert_not_called()


def test_cleanup_reconciles_resource_that_appears_after_initial_404(
        monkeypatch):
    clock = [0.0]
    state = {
        'visible': False,
        'presence_checks': 0,
        'late_deletes': 0,
    }

    def delete_pass(cls, *args):
        del cls, args
        if state['visible']:
            state['visible'] = False
            state['late_deletes'] += 1
            return True
        return False

    def resources_present(cls, *args):
        del cls, args
        state['presence_checks'] += 1
        # Simulate an accepted insert becoming visible only after cleanup's
        # first not-found result.
        if state['presence_checks'] == 2:
            state['visible'] = True
        return ['regional MIG'] if state['visible'] else []

    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_delete_tpu_mig_resources', classmethod(delete_pass))
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_tpu_mig_resources_present',
                        classmethod(resources_present))
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_TPU_MIG_CLEANUP_RECONCILIATION_SECONDS', 3)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_TPU_MIG_CLEANUP_POLL_INTERVAL_SECONDS', 1)
    monkeypatch.setattr(instance_utils.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(
        instance_utils.time, 'sleep',
        lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    deleted = (instance_utils.GCPManagedInstanceGroup.delete_tpu_mig_resources(
        'project', 'us-east5', 'cluster'))

    assert deleted
    assert state['late_deletes'] == 1
    assert state['presence_checks'] >= 3


def test_cleanup_accepts_delete_error_after_absence_is_verified(monkeypatch):
    clock = [0.0]

    def delete_pass(cls, *args):
        del cls, args
        raise RuntimeError('redundant delete conflict')

    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_delete_tpu_mig_resources', classmethod(delete_pass))
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_tpu_mig_resources_present',
                        classmethod(lambda cls, *args: []))
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_TPU_MIG_CLEANUP_RECONCILIATION_SECONDS', 2)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_TPU_MIG_CLEANUP_POLL_INTERVAL_SECONDS', 1)
    monkeypatch.setattr(instance_utils.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(
        instance_utils.time, 'sleep',
        lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    deleted = (instance_utils.GCPManagedInstanceGroup.delete_tpu_mig_resources(
        'project', 'us-east5', 'cluster'))

    assert not deleted
