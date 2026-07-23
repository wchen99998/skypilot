"""Tests for GCP TPU Flex-start MIG permission preflight."""

from unittest import mock

import pytest

from sky import skypilot_config
from sky.clouds.utils import gcp_utils
from sky.provision import common
from sky.provision.gcp import constants
from sky.provision.gcp import instance_utils
from sky.provision.gcp import mig_utils
from sky.utils import annotations


@pytest.fixture(autouse=True)
def clear_permission_cache():
    mig_utils.get_missing_tpu_flex_start_permissions.cache_clear()
    yield
    mig_utils.get_missing_tpu_flex_start_permissions.cache_clear()


def _get_config_side_effect(managed_instance_group):

    def get_config(cloud, region, keys, default_value=None, **kwargs):
        del cloud, region, kwargs
        if keys == ('managed_instance_group',):
            return managed_instance_group
        if keys == ('vpc_name',):
            return 'custom-vpc'
        return default_value

    return get_config


def test_minimal_permissions_include_tpu_flex_start_when_configured():
    with mock.patch.object(skypilot_config,
                           'get_effective_region_config',
                           side_effect=_get_config_side_effect(
                               {'run_duration': 3600})):
        permissions = gcp_utils.get_minimal_compute_permissions()

    assert set(constants.TPU_FLEX_START_MIG_PERMISSIONS).issubset(permissions)
    assert len(permissions) == len(set(permissions))


def test_minimal_permissions_skip_tpu_flex_start_when_not_configured():
    with mock.patch.object(skypilot_config,
                           'get_effective_region_config',
                           side_effect=_get_config_side_effect(None)):
        permissions = gcp_utils.get_minimal_compute_permissions()

    assert set(constants.TPU_FLEX_START_MIG_PERMISSIONS).isdisjoint(permissions)


def test_tpu_flex_start_permission_check_is_cached(monkeypatch):
    crm = mock.MagicMock()
    execute = (
        crm.projects.return_value.testIamPermissions.return_value.execute)
    execute.return_value = {
        'permissions': constants.TPU_FLEX_START_MIG_PERMISSIONS,
    }
    build = mock.MagicMock(return_value=crm)
    monkeypatch.setattr(mig_utils.gcp, 'build', build)

    assert mig_utils.get_missing_tpu_flex_start_permissions('project') == []
    assert mig_utils.get_missing_tpu_flex_start_permissions('project') == []

    build.assert_called_once_with('cloudresourcemanager',
                                  'v1',
                                  credentials=None,
                                  cache_discovery=False)
    crm.projects.return_value.testIamPermissions.assert_called_once_with(
        resource='project',
        body={
            'permissions': constants.TPU_FLEX_START_MIG_PERMISSIONS,
        })
    execute.assert_called_once_with(num_retries=5)


def test_tpu_flex_start_permission_cache_is_request_scoped(monkeypatch):
    crm = mock.MagicMock()
    execute = (
        crm.projects.return_value.testIamPermissions.return_value.execute)
    execute.side_effect = [{
        'permissions': constants.TPU_FLEX_START_MIG_PERMISSIONS,
    }, {
        'permissions': [],
    }]
    build = mock.MagicMock(return_value=crm)
    monkeypatch.setattr(mig_utils.gcp, 'build', build)

    assert mig_utils.get_missing_tpu_flex_start_permissions('project') == []

    # API-server workers clear request-scoped caches before installing the next
    # request's credential environment.
    annotations.clear_request_level_cache()

    assert mig_utils.get_missing_tpu_flex_start_permissions(
        'project') == sorted(constants.TPU_FLEX_START_MIG_PERMISSIONS)
    assert build.call_count == 2


def test_tpu_flex_start_permission_check_reports_sorted_missing(monkeypatch):
    crm = mock.MagicMock()
    execute = (
        crm.projects.return_value.testIamPermissions.return_value.execute)
    execute.return_value = {
        'permissions': [
            permission
            for permission in constants.TPU_FLEX_START_MIG_PERMISSIONS
            if permission not in {
                'compute.instanceGroups.delete',
                'compute.regionOperations.get',
                'compute.instanceGroupManagers.get',
            }
        ],
    }
    monkeypatch.setattr(mig_utils.gcp, 'build',
                        mock.MagicMock(return_value=crm))

    assert mig_utils.get_missing_tpu_flex_start_permissions('project') == [
        'compute.instanceGroupManagers.get',
        'compute.instanceGroups.delete',
        'compute.regionOperations.get',
    ]


def test_tpu_mig_create_fails_before_resource_calls_on_missing_permissions(
        monkeypatch):
    monkeypatch.setattr(mig_utils, 'get_missing_tpu_flex_start_permissions',
                        lambda project_id: ['compute.resourcePolicies.create'])
    get_template = mock.MagicMock()
    monkeypatch.setattr(mig_utils, 'get_region_instance_template', get_template)

    with pytest.raises(common.ProvisionerError) as exc_info:
        instance_utils.GCPManagedInstanceGroup.create_instances(
            cluster_name='cluster',
            project_id='project',
            zone='us-east5-a',
            node_config={
                constants.MANAGED_INSTANCE_GROUP_CONFIG: {
                    'run_duration': 3600,
                    'accelerator_topology': '1x1',
                },
                'machineType': 'ct6e-standard-1t',
            },
            labels={},
            count=1,
            total_count=1,
            include_head_node=True,
        )

    message = str(exc_info.value)
    assert 'compute.resourcePolicies.create' in message
    assert 'roles/compute.instanceAdmin.v1' in message
    assert 'roles/iam.serviceAccountUser' in message
    assert '`sky check`' in message
    get_template.assert_not_called()
