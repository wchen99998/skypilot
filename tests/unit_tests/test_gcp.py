import pathlib
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from sky import logs
from sky import resources
from sky import skypilot_config
from sky import task as task_lib
from sky.backends import backend_utils
from sky.catalog import gcp_catalog
from sky.clouds import cloud as cloud_lib
from sky.clouds import gcp as gcp_cloud
from sky.clouds import Region
from sky.clouds import Zone
from sky.clouds.gcp import GCP
from sky.clouds.utils import gcp_utils
from sky.jobs import controller as managed_job_controller
from sky.provision import common
from sky.provision.gcp import config as gcp_config
from sky.provision.gcp import constants as gcp_constants
from sky.provision.gcp import instance as gcp_instance
from sky.provision.gcp import instance_utils
from sky.provision.gcp import mig_utils
from sky.provision.gcp import volume_utils as gcp_volume_utils
from sky.utils import common_utils
from sky.utils import config_utils
from sky.utils import controller_utils
from sky.utils import resources_utils
from sky.utils import schemas
from sky.utils import status_lib


def test_gcp_rtxpro6000_instance_type_mapping():
    # RTXPRO6000 (GCP G4) maps to g4-standard-{48,96,192,384} for 1/2/4/8 GPUs.
    assert gcp_catalog._ACC_INSTANCE_TYPE_DICTS['RTXPRO6000'] == {
        1: ['g4-standard-48'],
        2: ['g4-standard-96'],
        4: ['g4-standard-192'],
        8: ['g4-standard-384'],
    }
    expected = {
        'g4-standard-48': 1,
        'g4-standard-96': 2,
        'g4-standard-192': 4,
        'g4-standard-384': 8,
    }
    for instance_type, count in expected.items():
        assert gcp_catalog._INSTANCE_TYPE_TO_ACC[instance_type] == {
            'RTXPRO6000': count,
        }


@pytest.mark.parametrize(
    'instance_type',
    ['g4-standard-48', 'g4-standard-96', 'g4-standard-192', 'g4-standard-384'])
def test_gcp_g4_uses_hyperdisk_balanced(instance_type):
    # G4 only supports hyperdisk-balanced (no pd-* support), like n4/a4.
    tier2name = gcp_volume_utils.get_data_disk_tier_mapping(instance_type)
    for tier in resources_utils.DiskTier:
        if tier == resources_utils.DiskTier.BEST:
            continue
        assert tier2name[tier] == 'hyperdisk-balanced', (
            f'{instance_type} tier {tier.value} should be hyperdisk-balanced, '
            f'got {tier2name[tier]}')


@pytest.mark.parametrize((
    'mock_return', 'expected'
), [([
    gcp_utils.GCPReservation(
        self_link=
        'https://www.googleapis.com/compute/v1/projects/<project>/zones/<zone>/reservations/<reservation>',
        specific_reservation=gcp_utils.SpecificReservation(count=1,
                                                           in_use_count=0),
        specific_reservation_required=True,
        zone='zone')
], {
    'projects/<project>/reservations/<reservation>': 1
}),
    ([
        gcp_utils.GCPReservation(
            self_link=
            'https://www.googleapis.com/compute/v1/projects/<project>/zones/<zone>/reservations/<reservation>',
            specific_reservation=gcp_utils.SpecificReservation(count=2,
                                                               in_use_count=1),
            specific_reservation_required=False,
            zone='zone')
    ], {
        'projects/<project>/reservations/<reservation>': 1
    }),
    ([
        gcp_utils.GCPReservation(
            self_link=
            'https://www.googleapis.com/compute/v1/projects/<project2>/zones/<zone>/reservations/<reservation>',
            specific_reservation=gcp_utils.SpecificReservation(count=1,
                                                               in_use_count=0),
            specific_reservation_required=True,
            zone='zone')
    ], {})])
def test_gcp_get_reservations_available_resources(mock_return, expected):
    gcp = GCP()
    with patch.object(gcp_utils,
                      'list_reservations_for_instance_type_in_zone',
                      return_value=mock_return):
        reservations = gcp.get_reservations_available_resources(
            'instance_type', 'region', 'zone',
            {'projects/<project>/reservations/<reservation>'})
        assert reservations == expected


def test_gcp_reservation_from_dict():
    r = gcp_utils.GCPReservation.from_dict({
        'selfLink': 'test',
        'specificReservation': {
            'count': '1',
            'inUseCount': '0'
        },
        'specificReservationRequired': True,
        'zone': 'zone'
    })

    assert r.self_link == 'test'
    assert r.specific_reservation.count == 1
    assert r.specific_reservation.in_use_count == 0
    assert r.specific_reservation_required == True
    assert r.zone == 'zone'


@pytest.mark.parametrize(('count', 'in_use_count', 'expected'), [(1, 0, 1),
                                                                 (1, 1, 0)])
def test_gcp_reservation_available_resources(count, in_use_count, expected):
    r = gcp_utils.GCPReservation(
        self_link='test',
        specific_reservation=gcp_utils.SpecificReservation(
            count=count, in_use_count=in_use_count),
        specific_reservation_required=True,
        zone='zone')

    assert r.available_resources == expected


def test_gcp_reservation_name():
    r = gcp_utils.GCPReservation(
        self_link=
        'https://www.googleapis.com/compute/v1/projects/<project>/zones/<zone>/reservations/<reservation-name>',
        specific_reservation=gcp_utils.SpecificReservation(count=1,
                                                           in_use_count=1),
        specific_reservation_required=True,
        zone='zone')
    assert r.name == 'projects/<project>/reservations/<reservation-name>'


def test_gcp_mig_instance_template_uses_flex_start(monkeypatch):
    compute = MagicMock()
    insert = compute.regionInstanceTemplates.return_value.insert
    insert.return_value.execute.return_value = {'name': 'operation'}
    monkeypatch.setattr(
        mig_utils.gcp,
        'build',
        lambda *args, **kwargs: compute,
    )

    node_config = {
        gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG: {
            'run_duration': 3600,
            'provision_timeout': 900,
        },
        'machineType': 'n1-standard-4',
        'guestAccelerators': [{
            'acceleratorType': 'nvidia-tesla-a100',
            'acceleratorCount': 1,
        }],
        'scheduling': {
            'onHostMaintenance': 'TERMINATE',
        },
        'description': 'Description returned by an existing template.',
        'reservationAffinity': {
            'consumeReservationType': 'SPECIFIC_RESERVATION',
        },
    }

    operation = mig_utils.create_region_instance_template(
        'cluster', 'project', 'us-central1', 'template', node_config)

    assert operation == {'name': 'operation'}
    insert.assert_called_once()
    body = insert.call_args.kwargs['body']
    properties = body['properties']
    assert gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG not in properties
    assert properties['description'] == (
        "SkyPilot instance template for 'cluster' to support DWS requests.")
    assert properties['reservationAffinity'] == {
        'consumeReservationType': 'NO_RESERVATION',
    }
    assert properties['scheduling'] == {
        'provisioningModel': 'FLEX_START',
        'instanceTerminationAction': 'DELETE',
        'maxRunDuration': {
            'seconds': '3600',
        },
        'onHostMaintenance': 'TERMINATE',
    }


def test_gcp_query_mig_instances_uses_group_membership(monkeypatch):
    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        lambda *args: ['cluster-head', 'cluster-worker'])
    filter_calls = []

    def filter_instances(cls,
                         project_id,
                         zone,
                         label_filters,
                         status_filters,
                         included_instances=None,
                         excluded_instances=None):
        filter_calls.append((project_id, zone, label_filters, status_filters,
                             included_instances, excluded_instances))
        return {
            'cluster-head': {
                'status': 'RUNNING',
            },
            'cluster-worker': {
                'status': 'RUNNING',
            },
        }

    monkeypatch.setattr(instance_utils.GCPComputeInstance, 'filter',
                        classmethod(filter_instances))

    statuses = gcp_instance.query_instances(
        cluster_name='display-name',
        cluster_name_on_cloud='cluster',
        provider_config={
            'availability_zone': 'us-east5-a',
            'project_id': 'project',
            'use_managed_instance_group': True,
        })

    assert statuses == {
        'cluster-head': (status_lib.ClusterStatus.UP, None),
        'cluster-worker': (status_lib.ClusterStatus.UP, None),
    }
    assert filter_calls == [('project', 'us-east5-a', None, None,
                             ['cluster-head', 'cluster-worker'], None)]


def test_gcp_query_missing_mig_returns_no_instances(monkeypatch):
    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        lambda *args: [])
    filter_instances = MagicMock()
    monkeypatch.setattr(instance_utils.GCPComputeInstance, 'filter',
                        filter_instances)

    statuses = gcp_instance.query_instances(
        cluster_name='display-name',
        cluster_name_on_cloud='cluster',
        provider_config={
            'availability_zone': 'us-east5-a',
            'project_id': 'project',
            'use_managed_instance_group': True,
        })

    assert statuses == {}
    filter_instances.assert_not_called()


def test_gcp_list_regional_mig_instances_paginates(monkeypatch):
    compute = MagicMock()
    regional_list = (
        compute.regionInstanceGroupManagers.return_value.listManagedInstances)
    regional_list.return_value.execute.side_effect = [{
        'managedInstances': [{
            'name': 'cluster-head',
        }],
        'nextPageToken': 'next-page',
    }, {
        'managedInstances': [{
            'name': 'cluster-worker',
        }],
    }]
    monkeypatch.setattr(mig_utils.gcp, 'build', lambda *args, **kwargs: compute)

    instances = mig_utils.list_managed_instance_group_instances(
        'project', 'us-east5-a', 'sky-mig-cluster')

    assert instances == ['cluster-head', 'cluster-worker']
    assert regional_list.call_args_list == [
        call(project='project',
             region='us-east5',
             instanceGroupManager='sky-mig-cluster'),
        call(project='project',
             region='us-east5',
             instanceGroupManager='sky-mig-cluster',
             pageToken='next-page'),
    ]


def test_gcp_list_zonal_mig_instances_after_regional_not_found(monkeypatch):

    class FakeHttpError(Exception):
        pass

    compute = MagicMock()
    regional_execute = (compute.regionInstanceGroupManagers.return_value.
                        listManagedInstances.return_value.execute)
    regional_execute.side_effect = FakeHttpError(
        "The resource 'projects/project/regions/us-east5/"
        "instanceGroupManagers/sky-mig-cluster' was not found")
    zonal_list = compute.instanceGroupManagers.return_value.listManagedInstances
    zonal_list.return_value.execute.return_value = {
        'managedInstances': [{
            'name': 'cluster-head',
        }],
    }
    monkeypatch.setattr(mig_utils.gcp, 'build', lambda *args, **kwargs: compute)
    monkeypatch.setattr(mig_utils.gcp, 'http_error_exception',
                        lambda: FakeHttpError)

    instances = mig_utils.list_managed_instance_group_instances(
        'project', 'us-east5-a', 'sky-mig-cluster')

    assert instances == ['cluster-head']
    zonal_list.assert_called_once_with(project='project',
                                       zone='us-east5-a',
                                       instanceGroupManager='sky-mig-cluster')


def test_gcp_list_missing_mig_returns_no_instances(monkeypatch):

    class FakeHttpError(Exception):
        pass

    compute = MagicMock()
    regional_execute = (compute.regionInstanceGroupManagers.return_value.
                        listManagedInstances.return_value.execute)
    regional_execute.side_effect = FakeHttpError(
        "The resource 'projects/project/regions/us-east5/"
        "instanceGroupManagers/sky-mig-cluster' was not found")
    zonal_execute = (compute.instanceGroupManagers.return_value.
                     listManagedInstances.return_value.execute)
    zonal_execute.side_effect = FakeHttpError(
        "The resource 'projects/project/zones/us-east5-a/"
        "instanceGroupManagers/sky-mig-cluster' was not found")
    monkeypatch.setattr(mig_utils.gcp, 'build', lambda *args, **kwargs: compute)
    monkeypatch.setattr(mig_utils.gcp, 'http_error_exception',
                        lambda: FakeHttpError)

    instances = mig_utils.list_managed_instance_group_instances(
        'project', 'us-east5-a', 'sky-mig-cluster')

    assert instances == []


@pytest.mark.parametrize('instance_type',
                         ['ct6e-standard-8t', 'tpu7x-standard-4t'])
def test_gcp_compute_tpu_machine_type_uses_mig_without_guest_accelerators(
        instance_type):
    assert instance_utils.get_node_type({
        gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG: {
            'run_duration': 3600,
            'provision_timeout': 900,
        },
        'machineType': instance_type,
    }) == instance_utils.GCPNodeType.MIG


def test_gcp_cpu_machine_type_with_mig_config_uses_compute():
    assert instance_utils.get_node_type({
        gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG: {
            'run_duration': 3600,
            'provision_timeout': 900,
        },
        'machineType': 'n1-standard-4',
    }) == instance_utils.GCPNodeType.COMPUTE


@pytest.mark.parametrize('instance_type',
                         ['ct6e-standard-4t', 'tpu7x-standard-4t'])
def test_gcp_compute_tpu_boot_disk_uses_hyperdisk_balanced(instance_type):
    for disk_tier in (
            None,
            resources_utils.DiskTier.LOW,
            resources_utils.DiskTier.MEDIUM,
            resources_utils.DiskTier.HIGH,
            resources_utils.DiskTier.ULTRA,
            resources_utils.DiskTier.BEST,
    ):
        assert GCP._get_disk_type(instance_type,
                                  disk_tier) == 'hyperdisk-balanced'


def test_gcp_compute_tpu_mig_disables_stop_and_spot():
    resource = resources.Resources(cloud=GCP(),
                                   instance_type='tpu7x-standard-4t',
                                   region='us-central1',
                                   _cluster_config_overrides={
                                       'gcp': {
                                           'managed_instance_group': {
                                               'run_duration': 3600,
                                               'accelerator_topology': '2x2x1',
                                           },
                                       },
                                   })

    unsupported = GCP._unsupported_features_for_resources(resource)

    assert cloud_lib.CloudImplementationFeatures.STOP in unsupported
    assert cloud_lib.CloudImplementationFeatures.SPOT_INSTANCE in unsupported


def test_gcp_mig_resource_names_fit_gce_limit():
    cluster_name = 'spectra-' + 'long-' * 20

    for resource_name in (
            mig_utils.get_instance_template_name(cluster_name),
            mig_utils.get_managed_instance_group_name(cluster_name),
            mig_utils.get_workload_policy_name(cluster_name),
    ):
        assert len(resource_name) <= mig_utils.GCE_RESOURCE_NAME_MAX_LENGTH
        assert resource_name.startswith('sky-')


def test_gcp_tpu_mig_compute_api_requests(monkeypatch):
    compute = MagicMock()
    workload_operation = {'name': 'workload-operation'}
    mig_operation = {'name': 'mig-operation'}
    compute.resourcePolicies.return_value.insert.return_value.execute.return_value = (
        workload_operation)
    compute.regionInstanceGroupManagers.return_value.insert.return_value.execute.return_value = (
        mig_operation)
    monkeypatch.setattr(mig_utils.gcp, 'build', lambda *args, **kwargs: compute)

    assert mig_utils.create_workload_policy(
        'project', 'us-east5', 'policy', '4x8',
        'AUTO_CONNECT') == workload_operation
    assert mig_utils.create_region_managed_instance_group(
        'project',
        'us-east5',
        ['us-east5-a'],
        'mig',
        'projects/project/regions/us-east5/instanceTemplates/template',
        8,
        'projects/project/regions/us-east5/resourcePolicies/policy',
    ) == mig_operation

    compute.resourcePolicies.return_value.insert.assert_called_once_with(
        project='project',
        region='us-east5',
        body={
            'name': 'policy',
            'workloadPolicy': {
                'type': 'HIGH_THROUGHPUT',
                'acceleratorTopology': '4x8',
                'acceleratorTopologyMode': 'AUTO_CONNECT',
            },
        })
    compute.regionInstanceGroupManagers.return_value.insert.assert_called_once_with(
        project='project',
        region='us-east5',
        body={
            'name': 'mig',
            'instanceTemplate':
                ('projects/project/regions/us-east5/instanceTemplates/'
                 'template'),
            'targetSize': 8,
            'targetSizePolicy': {
                'mode': 'BULK',
            },
            'distributionPolicy': {
                'targetShape': 'ANY_SINGLE_ZONE',
                'zones': [{
                    'zone': 'projects/project/zones/us-east5-a',
                }],
            },
            'instanceLifecyclePolicy': {
                'defaultActionOnFailure': 'DO_NOTHING',
            },
            'resourcePolicies': {
                'workloadPolicy':
                    ('projects/project/regions/us-east5/resourcePolicies/'
                     'policy'),
            },
            'updatePolicy': {
                'type': 'OPPORTUNISTIC',
                'instanceRedistributionType': 'NONE',
            },
        })


def test_gcp_tpu_mig_compute_api_waits_through_capacity_error(monkeypatch):
    compute = MagicMock()
    sleep_calls = []
    execute = (compute.regionInstanceGroupManagers.return_value.get.
               return_value.execute)
    execute.side_effect = [{
        'targetSize': 2,
        'status': {
            'isStable': False,
            'bulkInstanceOperation': {
                'inProgress': True,
                'lastProgressCheck': {
                    'timestamp': '2026-07-22T01:00:00Z',
                    'error': {
                        'errors': [{
                            'code': 'ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS',
                            'message': 'Waiting for resources.',
                        }],
                    },
                },
            },
            'currentInstanceStatuses': {
                'pending': 2,
                'running': 0,
            },
        },
    }, {
        'targetSize': 2,
        'status': {
            'isStable': True,
            'bulkInstanceOperation': {
                'inProgress': False,
            },
            'currentInstanceStatuses': {
                'pending': 0,
                'running': 2,
            },
        },
    }]
    monkeypatch.setattr(mig_utils.gcp, 'build', lambda *args, **kwargs: compute)
    monkeypatch.setattr(mig_utils.time, 'sleep', sleep_calls.append)

    mig_utils.wait_for_region_managed_group_to_be_stable('project',
                                                         'us-east5',
                                                         'mig',
                                                         timeout=86400)

    assert execute.call_count == 2
    assert sleep_calls == [mig_utils._BULK_MIG_POLL_INTERVAL_SECONDS]


def test_gcp_tpu_mig_compute_api_wait_timeout(monkeypatch):
    compute = MagicMock()
    execute = (compute.regionInstanceGroupManagers.return_value.get.
               return_value.execute)
    execute.return_value = {
        'targetSize': 2,
        'status': {
            'isStable': False,
            'bulkInstanceOperation': {
                'inProgress': True,
                'lastProgressCheck': {
                    'error': {
                        'errors': [{
                            'code': 'QUOTA_EXCEEDED',
                            'message': 'Quota is insufficient.',
                        }],
                    },
                },
            },
            'currentInstanceStatuses': {
                'pending': 2,
            },
        },
    }
    monkeypatch.setattr(mig_utils.gcp, 'build', lambda *args, **kwargs: compute)
    monotonic_values = iter([0.0, 2.0])
    monkeypatch.setattr(mig_utils.time, 'monotonic',
                        lambda: next(monotonic_values))

    with pytest.raises(TimeoutError, match='QUOTA_EXCEEDED'):
        mig_utils.wait_for_region_managed_group_to_be_stable('project',
                                                             'us-east5',
                                                             'mig',
                                                             timeout=1)


def test_gcp_tpu_mig_waits_for_running_status_after_stable(monkeypatch):
    compute = MagicMock()
    execute = (compute.regionInstanceGroupManagers.return_value.get.
               return_value.execute)
    execute.side_effect = [{
        'targetSize': 1,
        'status': {
            'isStable': True,
            'bulkInstanceOperation': {
                'inProgress': False,
            },
        },
    }, {
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
    }]
    monkeypatch.setattr(mig_utils.gcp, 'build', lambda *args, **kwargs: compute)
    monkeypatch.setattr(mig_utils.time, 'sleep', lambda _: None)

    mig_utils.wait_for_region_managed_group_to_be_stable('project',
                                                         'us-east5',
                                                         'mig',
                                                         timeout=60)

    assert execute.call_count == 2


@pytest.mark.parametrize(('machine_type', 'zone', 'topology', 'num_nodes'), [
    ('ct6e-standard-1t', 'us-east5-a', '1x1', 1),
    ('ct6e-standard-4t', 'us-east5-a', '2x2', 1),
    ('ct6e-standard-4t', 'us-east5-a', '4x8', 8),
    ('ct6e-standard-8t-tpu', 'us-east5-a', '2x4', 1),
    ('tpu7x-standard-4t', 'us-central1-c', '2x2x1', 1),
    ('tpu7x-standard-4t-tpu', 'us-central1-c', '2x4x4', 8),
])
def test_gcp_compute_tpu_flex_start_config_validates(machine_type, zone,
                                                     topology, num_nodes):
    mig_utils.validate_tpu_flex_start_config(machine_type, zone, topology,
                                             num_nodes)


@pytest.mark.parametrize(
    ('machine_type', 'zone', 'topology', 'num_nodes', 'error'), [
        ('ct6e-standard-4t', 'us-east5-b', '4x8', 8,
         'does not support TPU Flex-start'),
        ('ct6e-standard-4t', 'us-east5-a', '2x2x2', 2,
         'requires a 2D accelerator topology'),
        ('ct6e-standard-4t', 'us-east5-a', '2x4', 2,
         'not supported .* on Compute Engine'),
        ('ct6e-standard-4t', 'us-east5-a', '1x4', 1,
         'not supported .* on Compute Engine'),
        ('tpu7x-standard-4t', 'us-central1-c', '4x8', 8,
         'requires a 3D accelerator topology'),
        ('tpu7x-standard-4t', 'us-central1-c', '1x1x4', 1,
         'not supported .* on Compute Engine'),
        ('tpu7x-standard-4t', 'us-central1-c', '2x2x2', 1, 'requires 2 .* VMs'),
    ])
def test_gcp_compute_tpu_flex_start_config_rejects_mismatch(
        machine_type, zone, topology, num_nodes, error):
    with pytest.raises(ValueError, match=error):
        mig_utils.validate_tpu_flex_start_config(machine_type, zone, topology,
                                                 num_nodes)


def test_gcp_ct6e_mig_uses_regional_bulk_workload_policy(monkeypatch):
    calls = []

    def add_labels_and_find_head(cls, *args, **kwargs):
        del cls
        calls.append(('add-labels', args, kwargs))
        return ['node-0', 'node-1', 'node-2', 'node-3']

    monkeypatch.setattr(mig_utils, 'check_instance_template_exits',
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(mig_utils, 'check_region_managed_instance_group_exists',
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(
        mig_utils, 'check_managed_instance_group_exists',
        lambda *args, **kwargs:
        (_ for _ in
         ()).throw(AssertionError('TPU MIG should not use zonal MIG lookup')))
    monkeypatch.setattr(mig_utils, 'check_workload_policy_exists',
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(
        mig_utils, 'delete_workload_policy',
        lambda *args, **kwargs: calls.append(
            ('delete-workload-policy', args, kwargs)))
    monkeypatch.setattr(
        mig_utils, 'create_region_instance_template',
        lambda *args, **kwargs: calls.append(
            ('template', args, kwargs)) or {'name': 'template-op'})
    monkeypatch.setattr(
        mig_utils, 'create_workload_policy',
        lambda *args, **kwargs: calls.append(('workload-policy', args, kwargs)))
    monkeypatch.setattr(
        mig_utils, 'create_region_managed_instance_group',
        lambda *args, **kwargs: calls.append(('regional-mig', args, kwargs)))
    monkeypatch.setattr(
        mig_utils, 'wait_for_region_managed_group_to_be_stable',
        lambda *args, **kwargs: calls.append(
            ('wait-regional-mig', args, kwargs)))
    monkeypatch.setattr(
        mig_utils, 'create_managed_instance_group', lambda *args, **kwargs:
        (_ for _ in
         ()).throw(AssertionError('TPU MIG should not use zonal MIG creation')))
    monkeypatch.setattr(
        mig_utils, 'resize_managed_instance_group', lambda *args, **kwargs:
        (_ for _ in
         ()).throw(AssertionError('TPU MIG should not use resize requests')))
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, 'wait_for_operation',
        classmethod(lambda cls, *args, **kwargs: calls.append(
            ('wait-operation', args, kwargs))))
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_add_labels_and_find_head',
                        classmethod(add_labels_and_find_head))
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, 'create_node_tag',
        classmethod(lambda cls, *args, **kwargs: calls.append(
            ('head-tag', args, kwargs))))

    _, instance_names = instance_utils.GCPManagedInstanceGroup.create_instances(
        cluster_name='cluster',
        project_id='project',
        zone='us-east5-a',
        node_config={
            gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG: {
                'run_duration': 3600,
                'provision_timeout': 900,
                'accelerator_topology': '4x4',
                'accelerator_topology_mode': 'AUTO_CONNECT',
            },
            'machineType': 'ct6e-standard-4t',
            'labels': {
                'ray-cluster-name': 'stale-cluster',
                'skypilot-cluster-name': 'stale-cluster',
            },
        },
        labels={},
        count=4,
        total_count=4,
        include_head_node=True,
    )

    assert instance_names == ['node-0', 'node-1', 'node-2', 'node-3']
    assert [call[0] for call in calls] == [
        'delete-workload-policy',
        'template',
        'wait-operation',
        'workload-policy',
        'wait-operation',
        'regional-mig',
        'wait-operation',
        'wait-regional-mig',
        'add-labels',
        'head-tag',
    ]
    assert calls[0][1] == ('project', 'us-east5', 'sky-wp-cluster')
    regional_mig_call = calls[5]
    assert regional_mig_call[2] == {
        'size': 4,
        'workload_policy_url':
            'projects/project/regions/us-east5/resourcePolicies/'
            'sky-wp-cluster',
    }
    assert regional_mig_call[1][:5] == (
        'project',
        'us-east5',
        ['us-east5-a'],
        'sky-mig-cluster',
        'projects/project/regions/us-east5/instanceTemplates/sky-it-cluster',
    )
    relabel_call = calls[-2]
    assert relabel_call[1][3] == {
        'ray-cluster-name': 'cluster',
        'ray-node-type': 'worker',
        'skypilot-cluster-name': 'cluster',
        'skypilot-head-node': '0',
    }


def test_gcp_ct6e_mig_cleans_up_created_resources_on_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(mig_utils, 'check_instance_template_exits',
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(mig_utils, 'check_region_managed_instance_group_exists',
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(
        mig_utils, 'check_managed_instance_group_exists',
        lambda *args, **kwargs:
        (_ for _ in
         ()).throw(AssertionError('TPU MIG should not use zonal MIG lookup')))
    monkeypatch.setattr(mig_utils, 'check_workload_policy_exists',
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(
        mig_utils, 'create_region_instance_template',
        lambda *args, **kwargs: calls.append(
            ('template', args, kwargs)) or {'name': 'template-op'})
    monkeypatch.setattr(
        mig_utils, 'create_workload_policy',
        lambda *args, **kwargs: calls.append(('workload-policy', args, kwargs)))
    monkeypatch.setattr(
        mig_utils, 'create_region_managed_instance_group',
        lambda *args, **kwargs: calls.append(('regional-mig', args, kwargs)))
    monkeypatch.setattr(
        mig_utils, 'wait_for_region_managed_group_to_be_stable',
        lambda *args, **kwargs:
        (_ for _ in ()).throw(RuntimeError('quota exhausted')))
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, 'wait_for_operation',
        classmethod(lambda cls, *args, **kwargs: calls.append(
            ('wait-operation', args, kwargs))))
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, '_delete_tpu_mig_resources',
        classmethod(lambda cls, *args, **kwargs: calls.append(
            ('cleanup', args, kwargs))))

    with pytest.raises(RuntimeError, match='quota exhausted'):
        instance_utils.GCPManagedInstanceGroup.create_instances(
            cluster_name='cluster',
            project_id='project',
            zone='us-south1-ai1b',
            node_config={
                gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG: {
                    'run_duration': 3600,
                    'provision_timeout': 900,
                    'accelerator_topology': '4x4',
                    'accelerator_topology_mode': 'AUTO_CONNECT',
                },
                'machineType': 'ct6e-standard-4t',
                'labels': {},
            },
            labels={},
            count=4,
            total_count=4,
            include_head_node=True,
        )

    assert [call[0] for call in calls] == [
        'template',
        'wait-operation',
        'workload-policy',
        'wait-operation',
        'regional-mig',
        'wait-operation',
        'cleanup',
    ]
    assert calls[-1][1] == ('project', 'us-south1', 'cluster')


@pytest.mark.parametrize('mig_exists', [False, True])
def test_gcp_tpu_mig_cleanup_deletes_all_resources(monkeypatch, mig_exists):
    calls = []
    monkeypatch.setattr(
        mig_utils, 'check_region_managed_instance_group_exists',
        lambda *args, **kwargs: calls.append(('check-mig', args)) or mig_exists)
    monkeypatch.setattr(
        mig_utils, 'delete_region_managed_instance_group',
        lambda *args, **kwargs: calls.append(('delete-mig', args)) or {
            'name': 'delete-mig-op',
        })
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, 'wait_for_operation',
        classmethod(lambda cls, *args, **kwargs: calls.append(
            ('wait-operation', args, kwargs))))
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup,
        '_delete_region_instance_template',
        classmethod(lambda cls, *args, **kwargs: calls.append(
            ('delete-template', args))))
    monkeypatch.setattr(
        mig_utils, 'delete_workload_policy',
        lambda *args, **kwargs: calls.append(('delete-policy', args)))

    deleted = instance_utils.GCPManagedInstanceGroup.delete_tpu_mig_resources(
        'project', 'us-east5', 'cluster')

    assert deleted is mig_exists
    expected_calls = [('check-mig', ('project', 'us-east5', 'sky-mig-cluster'))]
    if mig_exists:
        expected_calls.extend([
            ('delete-mig', ('project', 'us-east5', 'sky-mig-cluster')),
            ('wait-operation', ({
                'name': 'delete-mig-op',
            }, 'project'), {
                'region': 'us-east5',
            }),
        ])
    expected_calls.extend([
        ('delete-template', ('project', 'us-east5', 'sky-it-cluster')),
        ('delete-policy', ('project', 'us-east5', 'sky-wp-cluster')),
    ])
    assert calls == expected_calls


def test_gcp_delete_mig_cleans_tpu_policy_after_mig_expired(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mig_utils, 'check_region_managed_instance_group_exists',
        lambda *args, **kwargs: calls.append(('check-mig', args)) or False)
    monkeypatch.setattr(
        mig_utils, 'check_workload_policy_exists',
        lambda *args, **kwargs: calls.append(('check-policy', args)) or True)
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, '_delete_tpu_mig_resources',
        classmethod(lambda cls, *args, **kwargs: calls.append(
            ('delete-tpu-resources', args)) or False))
    monkeypatch.setattr(
        mig_utils, 'cancel_all_resize_request_for_mig', lambda *args, **kwargs:
        (_ for _ in ()).throw(
            AssertionError('TPU cleanup must not use zonal resize requests')))

    deleted = instance_utils.GCPManagedInstanceGroup.delete_mig(
        'project', 'us-east5-a', 'cluster')

    assert deleted is False
    assert calls == [
        ('check-mig', ('project', 'us-east5', 'sky-mig-cluster')),
        ('check-policy', ('project', 'us-east5', 'sky-wp-cluster')),
        ('delete-tpu-resources', ('project', 'us-east5', 'cluster')),
    ]


@pytest.mark.parametrize(('instance_type', 'region'), [
    ('ct6e-standard-4t', 'us-south1'),
    ('tpu7x-standard-4t', 'us-central1'),
])
def test_managed_job_cleanup_deletes_gcp_tpu_mig_orphan(monkeypatch,
                                                        instance_type, region):
    calls = []
    monkeypatch.setattr(GCP, 'get_project_id',
                        classmethod(lambda cls, dryrun=False: 'project'))
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, 'delete_tpu_mig_resources',
        classmethod(lambda cls, *args, **kwargs: calls.append((args, kwargs))))

    task = task_lib.Task(name='train')
    task_resources = resources.Resources(
        cloud=GCP(),
        instance_type=instance_type,
        region=region,
        _cluster_config_overrides={
            'gcp': {
                'managed_instance_group': {
                    'run_duration': 3600,
                    'provision_timeout': 900,
                    'accelerator_topology': '4x4',
                    'accelerator_topology_mode': 'AUTO_CONNECT',
                },
            },
        },
    )
    task.set_resources({task_resources})
    task.best_resources = task_resources

    cluster_name = 'spectra-300m-h14-noint-7i-6'
    managed_job_controller._cleanup_gcp_tpu_mig_if_needed(task, cluster_name)

    assert calls == [(
        (
            'project',
            region,
            common_utils.make_cluster_name_on_cloud(
                cluster_name, max_length=GCP().max_cluster_name_length()),
        ),
        {},
    )]


def test_gcp_ct6e_mig_requires_accelerator_topology(monkeypatch):
    monkeypatch.setattr(mig_utils, 'check_instance_template_exits',
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(mig_utils, 'check_region_managed_instance_group_exists',
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, 'filter',
        classmethod(lambda cls, *args, **kwargs: {'node-0': {}}))

    with pytest.raises(common.ProvisionerError, match='accelerator_topology'):
        instance_utils.GCPManagedInstanceGroup.create_instances(
            cluster_name='cluster',
            project_id='project',
            zone='us-east5-a',
            node_config={
                gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG: {
                    'run_duration': 3600,
                },
                'machineType': 'ct6e-standard-4t',
                'labels': {},
            },
            labels={},
            count=1,
            total_count=1,
            include_head_node=True,
        )


def test_gcp_ct6e_catalog_entries_use_documented_regions():
    assert gcp_catalog.instance_type_exists('ct6e-standard-8t') is True
    assert gcp_catalog.get_vcpus_mem_from_instance_type('ct6e-standard-8t') == (
        360, 1440)

    regions = gcp_catalog.get_region_zones_for_instance_type('ct6e-standard-8t',
                                                             use_spot=False)

    expected_region_to_zones = {
        'us-central1': ('us-central1-b',),
        'us-east1': ('us-east1-d',),
        'us-east5': ('us-east5-a', 'us-east5-b'),
        'us-south1': ('us-south1-ai1b',),
        'asia-northeast1': ('asia-northeast1-b',),
        'europe-west4': ('europe-west4-a',),
        'southamerica-west1': ('southamerica-west1-a',),
    }
    assert [region.name for region in regions] == list(expected_region_to_zones)
    assert {
        region.name: tuple(zone.name for zone in region.zones)
        for region in regions
    } == expected_region_to_zones


@pytest.mark.parametrize(('instance_type', 'expected_region_to_zones'), [
    ('ct6e-standard-4t', {
        'us-east5': ('us-east5-a',),
        'us-south1': ('us-south1-ai1b',),
        'asia-northeast1': ('asia-northeast1-b',),
    }),
    ('tpu7x-standard-4t', {
        'us-central1': ('us-central1-c',),
    }),
])
def test_gcp_compute_tpu_mig_offerings_use_flex_start_zones(
        instance_type, expected_region_to_zones):
    resource = resources.Resources(cloud=GCP(),
                                   instance_type=instance_type,
                                   _cluster_config_overrides={
                                       'gcp': {
                                           'managed_instance_group': {
                                               'run_duration': 3600,
                                               'accelerator_topology': '2x4',
                                           },
                                       },
                                   })

    regions = GCP.regions_with_offering(instance_type,
                                        accelerators=None,
                                        use_spot=False,
                                        region=None,
                                        zone=None,
                                        resources=resource)

    assert {
        region.name: tuple(zone.name for zone in region.zones)
        for region in regions
    } == expected_region_to_zones


def test_gcp_ct6e_ai_zone_validates():
    assert gcp_catalog.validate_region_zone(
        'us-south1', 'us-south1-ai1b') == ('us-south1', 'us-south1-ai1b')
    assert gcp_catalog.validate_region_zone(
        None, 'us-south1-ai1b') == ('us-south1', 'us-south1-ai1b')
    with pytest.raises(ValueError, match='Invalid zone'):
        gcp_catalog.validate_region_zone('us-east5', 'us-south1-ai1b')


def test_gcp_ct6e_catalog_uses_tpu_v6e_price(monkeypatch):
    calls = []

    def mock_accelerator_hourly_cost(accelerator,
                                     count,
                                     use_spot=False,
                                     region=None,
                                     zone=None):
        calls.append((accelerator, count, use_spot, region, zone))
        return 42.0

    monkeypatch.setattr(gcp_catalog, 'get_accelerator_hourly_cost',
                        mock_accelerator_hourly_cost)

    assert gcp_catalog.get_hourly_cost('ct6e-standard-8t',
                                       use_spot=False,
                                       region='us-east5',
                                       zone='us-east5-a') == 42.0
    assert calls == [('tpu-v6e-8', 1, False, 'us-east5', None)]


def test_gcp_ct6e_spot_is_not_launchable():
    assert gcp_catalog.get_region_zones_for_instance_type('ct6e-standard-8t',
                                                          use_spot=True) == []
    with pytest.raises(ValueError, match='do not support spot'):
        gcp_catalog.get_hourly_cost('ct6e-standard-8t', use_spot=True)


def test_gcp_tpu7x_catalog_entries():
    assert gcp_catalog.instance_type_exists('tpu7x-standard-4t') is True
    assert gcp_catalog.instance_type_exists('tpu7x-standard-4t-tpu') is True
    assert gcp_catalog.get_vcpus_mem_from_instance_type(
        'tpu7x-standard-4t') == (224, 960)
    assert gcp_catalog.get_hourly_cost('tpu7x-standard-4t',
                                       use_spot=False,
                                       region='us-central1',
                                       zone='us-central1-c') == 48.0

    regions = gcp_catalog.get_region_zones_for_instance_type(
        'tpu7x-standard-4t', use_spot=False)
    assert [region.name for region in regions] == ['us-central1']
    assert [zone.name for zone in regions[0].zones] == [
        'us-central1-ai1a',
        'us-central1-c',
    ]
    assert gcp_catalog.validate_region_zone(
        None, 'us-central1-ai1a') == ('us-central1', 'us-central1-ai1a')


def test_gcp_tpu7x_spot_is_not_launchable():
    assert gcp_catalog.get_region_zones_for_instance_type('tpu7x-standard-4t',
                                                          use_spot=True) == []
    with pytest.raises(ValueError, match='do not support spot'):
        gcp_catalog.get_hourly_cost('tpu7x-standard-4t', use_spot=True)


@pytest.mark.parametrize(
    ('instance_type', 'region_name', 'zone_name', 'image_family'), [
        ('ct6e-standard-4t', 'us-east5', 'us-east5-a',
         'ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e'),
        ('tpu7x-standard-4t', 'us-central1', 'us-central1-c',
         'ubuntu-accel-2404-amd64-tpu-tpu7x'),
    ])
def test_gcp_compute_tpu_uses_accelerator_image_by_default(
        instance_type, region_name, zone_name, image_family):
    resource = resources.Resources(cloud=GCP(),
                                   instance_type=instance_type,
                                   region=region_name)
    region = Region(region_name).set_zones([Zone(zone_name)])

    variables = GCP().make_deploy_resources_variables(
        resource,
        resources_utils.ClusterName('test', 'test'),
        region,
        region.zones,
        num_nodes=1,
        dryrun=True)

    assert variables['image_id'] == (
        'projects/ubuntu-os-accelerator-images/global/images/family/'
        f'{image_family}')
    assert variables['docker_run_options'] == ['--privileged']


def test_gcp_check_quota_skips_for_managed_instance_group(monkeypatch):
    """DWS/Flex-start should not be blocked by on-demand quota precheck."""
    resources_obj = resources.Resources(cloud=GCP(),
                                        region='us-central1',
                                        accelerators={'A100-80GB': 1})
    monkeypatch.setattr(
        skypilot_config,
        'get_effective_region_config',
        lambda *args, **kwargs: {
            'run_duration': 600,
        },
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError('Quota subprocess should not be called for DWS.')

    monkeypatch.setattr(gcp_cloud.subprocess_utils, 'run', fail_if_called)

    assert GCP.check_quota_available(resources_obj) is True


@pytest.mark.parametrize(
    ('specific_reservations', 'specific_reservation_required', 'expected'), [
        ([], False, True),
        ([], True, False),
        (['projects/<project>/reservations/<reservation>'], True, True),
        (['projects/<project>/reservations/<invalid>'], True, False),
    ])
def test_gcp_reservation_is_consumable(specific_reservations,
                                       specific_reservation_required, expected):
    r = gcp_utils.GCPReservation(
        self_link=
        'https://www.googleapis.com/compute/v1/projects/<project>/zones/<zone>/reservations/<reservation>',
        specific_reservation=gcp_utils.SpecificReservation(count=1,
                                                           in_use_count=1),
        specific_reservation_required=specific_reservation_required,
        zone='zone')
    assert r.is_consumable(
        specific_reservations=specific_reservations) is expected


def test_gcp_get_user_identities_workspace_cache_bypass():
    """Test that get_user_identities bypasses cache when workspace changes."""
    # Mock the external dependencies
    with patch('sky.clouds.gcp._run_output') as mock_run_output, \
         patch.object(GCP, 'get_project_id') as mock_get_project_id, \
         patch.object(skypilot_config, 'get_workspace_cloud') as mock_get_workspace_cloud:

        # Set up different project IDs for different workspaces
        def workspace_cloud_side_effect(cloud_name):
            current_workspace = skypilot_config.get_active_workspace()
            if current_workspace == 'default':
                return {'project_id': 'default-project'}
            elif current_workspace == 'other':
                return {'project_id': 'other-project'}
            return {}

        def project_id_side_effect():
            current_workspace = skypilot_config.get_active_workspace()
            if current_workspace == 'default':
                return 'default-project'
            elif current_workspace == 'other':
                return 'other-project'
            return 'fallback-project'

        mock_get_workspace_cloud.side_effect = workspace_cloud_side_effect
        mock_get_project_id.side_effect = project_id_side_effect
        mock_run_output.return_value = 'test@example.com'

        # First call in default workspace
        result1 = GCP.get_user_identities()
        expected1 = [['test@example.com [project_id=default-project]']]
        assert result1 == expected1

        # Switch to another workspace and call again
        with skypilot_config.local_active_workspace_ctx('other'):
            result2 = GCP.get_user_identities()
            expected2 = [['test@example.com [project_id=other-project]']]
            assert result2 == expected2

        # Back to default workspace - should get the original result
        result3 = GCP.get_user_identities()
        assert result3 == expected1

        # Verify that the underlying method was called for each different workspace config
        # Should be called 3 times total: once for default, once for other, once for default again
        assert mock_run_output.call_count == 2
        assert mock_get_project_id.call_count == 2

        # Verify workspace cloud was queried for each call
        assert mock_get_workspace_cloud.call_count == 3
        mock_get_workspace_cloud.assert_any_call('gcp')


def _make_subnet(name: str, vpc_name: str, project_id: str = 'test-project'):
    return {
        'name': name,
        'network': f'projects/{project_id}/global/networks/{vpc_name}',
        'selfLink': f'https://example.com/{name}',
    }


def _make_provision_config(provider_config):
    return common.ProvisionConfig(
        provider_config=provider_config,
        authentication_config={},
        docker_config={},
        node_config={},
        count=1,
        tags={},
        resume_stopped_nodes=False,
        ports_to_open_on_launch=None,
    )


def test_gcp_get_usable_vpc_and_subnet_uses_specified_subnet(monkeypatch):
    provider_config = {
        'project_id': 'test-project',
        'vpc_name': 'train-vpc',
        'subnet_names': ['train-subnet-b', 'train-subnet-a'],
    }
    provision_config = _make_provision_config(provider_config)
    monkeypatch.setattr(gcp_config, '_list_vpcnets', lambda *args, **kwargs: [{
        'name': 'train-vpc'
    }])
    monkeypatch.setattr(
        gcp_config, '_list_subnets', lambda *args, **kwargs: [
            _make_subnet('train-subnet-a', 'train-vpc'),
            _make_subnet('train-subnet-b', 'train-vpc'),
        ])

    vpc_name, subnet = gcp_config.get_usable_vpc_and_subnet(
        'cluster', 'us-central1', provision_config, MagicMock())

    assert vpc_name == 'train-vpc'
    assert subnet['name'] == 'train-subnet-b'


def test_gcp_get_usable_vpc_and_subnet_infers_vpc_from_subnet(monkeypatch):
    provider_config = {
        'project_id': 'test-project',
        'subnet_names': 'train-subnet',
    }
    provision_config = _make_provision_config(provider_config)
    monkeypatch.setattr(
        gcp_config, '_list_subnets', lambda *args, **kwargs: [
            _make_subnet('train-subnet', 'train-vpc'),
        ])

    vpc_name, subnet = gcp_config.get_usable_vpc_and_subnet(
        'cluster', 'us-central1', provision_config, MagicMock())

    assert vpc_name == 'train-vpc'
    assert subnet['name'] == 'train-subnet'


def test_gcp_get_usable_vpc_and_subnet_rejects_multiple_vpcs(monkeypatch):
    provider_config = {
        'project_id': 'test-project',
        'subnet_names': ['train-subnet-a', 'train-subnet-b'],
    }
    provision_config = _make_provision_config(provider_config)
    monkeypatch.setattr(
        gcp_config, '_list_subnets', lambda *args, **kwargs: [
            _make_subnet('train-subnet-a', 'train-vpc-a'),
            _make_subnet('train-subnet-b', 'train-vpc-b'),
        ])

    with pytest.raises(RuntimeError) as exc_info:
        gcp_config.get_usable_vpc_and_subnet('cluster', 'us-central1',
                                             provision_config, MagicMock())

    assert 'multiple VPCs' in str(exc_info.value)


def test_gcp_get_usable_vpc_and_subnet_partial_name_match(monkeypatch):
    provider_config = {
        'project_id': 'test-project',
        'vpc_name': 'train-vpc',
        'subnet_names': ['missing-subnet', 'train-subnet-b'],
    }
    provision_config = _make_provision_config(provider_config)
    monkeypatch.setattr(gcp_config, '_list_vpcnets', lambda *args, **kwargs: [{
        'name': 'train-vpc'
    }])
    monkeypatch.setattr(
        gcp_config, '_list_subnets', lambda *args, **kwargs: [
            _make_subnet('train-subnet-a', 'train-vpc'),
            _make_subnet('train-subnet-b', 'train-vpc'),
        ])

    vpc_name, subnet = gcp_config.get_usable_vpc_and_subnet(
        'cluster', 'us-central1', provision_config, MagicMock())

    assert vpc_name == 'train-vpc'
    assert subnet['name'] == 'train-subnet-b'


def test_gcp_get_usable_vpc_and_subnet_empty_subnet_names(monkeypatch):
    provider_config = {
        'project_id': 'test-project',
        'vpc_name': 'train-vpc',
        'subnet_names': [],
    }
    provision_config = _make_provision_config(provider_config)
    monkeypatch.setattr(gcp_config, '_list_vpcnets', lambda *args, **kwargs: [{
        'name': 'train-vpc'
    }])
    monkeypatch.setattr(
        gcp_config, '_list_subnets', lambda *args, **kwargs: [
            _make_subnet('train-subnet-a', 'train-vpc'),
            _make_subnet('train-subnet-b', 'train-vpc'),
        ])

    vpc_name, subnet = gcp_config.get_usable_vpc_and_subnet(
        'cluster', 'us-central1', provision_config, MagicMock())

    assert vpc_name == 'train-vpc'
    assert subnet['name'] == 'train-subnet-a'


def test_gcp_get_usable_vpc_and_subnet_shared_vpc_with_subnet_names(
        monkeypatch):
    provider_config = {
        'project_id': 'service-project',
        'vpc_name': 'host-project/train-vpc',
        'subnet_names': ['train-subnet-b'],
    }
    provision_config = _make_provision_config(provider_config)
    seen_projects = []

    def list_vpcnets(project_id, *args, **kwargs):
        seen_projects.append(project_id)
        return [{'name': 'train-vpc'}]

    def list_subnets(project_id, *args, **kwargs):
        seen_projects.append(project_id)
        return [
            _make_subnet('train-subnet-a',
                         'train-vpc',
                         project_id='host-project'),
            _make_subnet('train-subnet-b',
                         'train-vpc',
                         project_id='host-project'),
        ]

    monkeypatch.setattr(gcp_config, '_list_vpcnets', list_vpcnets)
    monkeypatch.setattr(gcp_config, '_list_subnets', list_subnets)

    vpc_name, subnet = gcp_config.get_usable_vpc_and_subnet(
        'cluster', 'us-central1', provision_config, MagicMock())

    assert seen_projects == ['host-project', 'host-project']
    assert vpc_name == 'train-vpc'
    assert subnet['name'] == 'train-subnet-b'
    assert subnet['network'] == (
        'projects/host-project/global/networks/train-vpc')


def test_gcp_minimal_compute_permissions_skip_firewall_for_custom_subnet():

    def get_effective_region_config_side_effect(cloud,
                                                region,
                                                keys,
                                                default_value=None,
                                                **kwargs):
        del cloud, region, kwargs
        if keys == ('subnet_names',):
            return ['train-subnet']
        return default_value

    with patch.object(skypilot_config,
                      'get_effective_region_config',
                      side_effect=get_effective_region_config_side_effect):
        permissions = gcp_utils.get_minimal_compute_permissions()

    for permission in gcp_constants.FIREWALL_PERMISSIONS:
        assert permission not in permissions


def test_gcp_minimal_compute_permissions_include_firewall_for_empty_subnets():

    def get_effective_region_config_side_effect(cloud,
                                                region,
                                                keys,
                                                default_value=None,
                                                **kwargs):
        del cloud, region, kwargs
        if keys == ('subnet_names',):
            return []
        return default_value

    with patch.object(skypilot_config,
                      'get_effective_region_config',
                      side_effect=get_effective_region_config_side_effect):
        permissions = gcp_utils.get_minimal_compute_permissions()

    for permission in gcp_constants.FIREWALL_PERMISSIONS:
        assert permission in permissions


def test_gcp_network_config_override_in_cluster_config(monkeypatch):
    """Test that GCP network overrides are passed through to the template."""
    monkeypatch.setattr(common_utils, 'make_cluster_name_on_cloud',
                        lambda *args, **kwargs: args[0])
    monkeypatch.setattr(backend_utils, '_get_yaml_path_from_cluster_name',
                        lambda *args, **kwargs: '/tmp/fake-gcp-yaml-path')
    monkeypatch.setattr(resources.Resources, 'make_deploy_variables',
                        lambda *args, **kwargs: {'region': 'us-central1'})
    monkeypatch.setattr(logs, 'get_logging_agent', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backend_utils.auth_utils, 'get_or_generate_keys',
        lambda *args, **kwargs:
        ('/tmp/fake-private-key', '/tmp/fake-public-key'))

    config_dict = config_utils.Config.from_dict({})
    monkeypatch.setattr(skypilot_config, '_get_loaded_config',
                        lambda *args, **kwargs: config_dict)

    override_configs = {
        'gcp': {
            'vpc_name': 'override-vpc',
            'subnet_names': ['override-subnet'],
        },
    }

    def fill_template_side_effect(*args, **kwargs):
        del kwargs
        template_vars = args[1]
        assert template_vars['vpc_name'] == 'override-vpc'
        assert template_vars['subnet_names'] == ['override-subnet']
        raise RuntimeError('fake-error')

    monkeypatch.setattr(common_utils, 'fill_template',
                        fill_template_side_effect)

    with pytest.raises(RuntimeError):
        backend_utils.write_cluster_config(
            to_provision=resources.Resources(
                cloud=GCP(),
                instance_type='n1-standard-4',
                _cluster_config_overrides=override_configs),
            num_nodes=1,
            cluster_config_template='gcp-ray.yml.j2',
            cluster_name='fake-gcp-cluster',
            local_wheel_path=pathlib.Path('fake-wheel-path'),
            wheel_hash='fake-wheel-hash',
            region=Region(name='us-central1'),
            zones=[Zone(name='us-central1-a')])


def test_gcp_controller_defaults_to_service_account_remote_identity(
        monkeypatch):
    monkeypatch.setattr(controller_utils.global_user_state,
                        'get_handle_from_cluster_name',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(skypilot_config, '_get_loaded_config',
                        lambda: config_utils.Config.from_dict({}))

    controller_resources = controller_utils.get_controller_resources(
        controller_utils.Controllers.JOBS_CONTROLLER,
        task_resources=[
            resources.Resources(cloud=GCP(),
                                instance_type='n1-standard-4',
                                region='us-central1')
        ])
    controller_resource = next(iter(controller_resources))

    assert skypilot_config.get_effective_workspace_region_config(
        cloud='gcp',
        region='us-central1',
        keys=('remote_identity',),
        default_value=None,
        override_configs=controller_resource.cluster_config_overrides,
    ) == schemas.RemoteIdentityOptions.SERVICE_ACCOUNT.value


def test_cloud_agnostic_controller_carries_gcp_service_account_default(
        monkeypatch):
    monkeypatch.setattr(controller_utils.global_user_state,
                        'get_handle_from_cluster_name',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(skypilot_config, '_get_loaded_config',
                        lambda: config_utils.Config.from_dict({}))

    controller_resources = controller_utils.get_controller_resources(
        controller_utils.Controllers.JOBS_CONTROLLER,
        task_resources=[resources.Resources(cpus='4+')])
    controller_resource = next(iter(controller_resources))

    assert controller_resource.cloud is None
    assert skypilot_config.get_effective_workspace_region_config(
        cloud='gcp',
        region=None,
        keys=('remote_identity',),
        default_value=None,
        override_configs=controller_resource.cluster_config_overrides,
    ) == schemas.RemoteIdentityOptions.SERVICE_ACCOUNT.value


def test_gcp_controller_respects_explicit_local_credentials(monkeypatch):
    monkeypatch.setattr(controller_utils.global_user_state,
                        'get_handle_from_cluster_name',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        skypilot_config, '_get_loaded_config',
        lambda: config_utils.Config.from_dict({
            'gcp': {
                'remote_identity': schemas.RemoteIdentityOptions.
                                   LOCAL_CREDENTIALS.value,
            },
        }))

    controller_resources = controller_utils.get_controller_resources(
        controller_utils.Controllers.JOBS_CONTROLLER,
        task_resources=[
            resources.Resources(cloud=GCP(),
                                instance_type='n1-standard-4',
                                region='us-central1')
        ])
    controller_resource = next(iter(controller_resources))

    assert skypilot_config.get_effective_workspace_region_config(
        cloud='gcp',
        region='us-central1',
        keys=('remote_identity',),
        default_value=None,
        override_configs=controller_resource.cluster_config_overrides,
    ) == schemas.RemoteIdentityOptions.LOCAL_CREDENTIALS.value


def test_gcp_template_does_not_mutate_remote_credentials(tmp_path):

    def render(remote_identity: str) -> str:
        output_path = tmp_path / f'gcp-{remote_identity}.yaml'
        common_utils.fill_template(
            'gcp-ray.yml.j2', {
                'cluster_name_on_cloud': 'test-cluster',
                'num_nodes': 1,
                'docker_image': None,
                'docker_run_options': [],
                'docker_login_config': None,
                'region': 'us-central1',
                'zones': 'us-central1-a',
                'gcp_project_id': 'test-project',
                'vpc_name': None,
                'subnet_names': None,
                'firewall_rule': None,
                'use_internal_ips': False,
                'force_enable_external_ips': False,
                'tpu_vm': False,
                'tpu_node_name': None,
                'gcp_use_managed_instance_group': False,
                'gcp_use_managed_instance_group_value': '',
                'enable_gvnic': None,
                'enable_gpu_direct': None,
                'placement_policy': None,
                'network_tier': None,
                'ssh_private_key': '/tmp/sky-key',
                'ssh_proxy_command': None,
                'user': 'test-user',
                'labels': {},
                'specific_reservations': [],
                'gpu': None,
                'machine_image': None,
                'instance_type': 'n1-standard-4',
                'disk_size': 50,
                'image_id': 'projects/test-project/global/images/test-image',
                'disk_tier': 'pd-balanced',
                'disk_iops': None,
                'volumes': [],
                'user_data': None,
                'use_spot': False,
                'gcp_queued_resource': None,
                'sky_ray_yaml_remote_path': '~/ray.yml',
                'sky_ray_yaml_local_path': '/tmp/ray.yml',
                'sky_remote_path': '~/.sky',
                'sky_wheel_hash': 'wheel-hash',
                'sky_local_path': '/tmp/sky-wheel',
                'credentials': {},
                'initial_setup_commands': [],
                'conda_installation_commands': '',
                'uv_installation_commands': '',
                'sky_pip_cmd': 'python -m pip',
                'ray_skypilot_installation_commands': '',
                'copy_skypilot_templates_commands': '',
                'ssh_max_sessions_config': '',
                'remote_identity': remote_identity,
            }, str(output_path))
        return output_path.read_text(encoding='utf-8')

    service_account_rendered = render(
        schemas.RemoteIdentityOptions.SERVICE_ACCOUNT.value)
    local_credentials_rendered = render(
        schemas.RemoteIdentityOptions.LOCAL_CREDENTIALS.value)

    for rendered in (service_account_rendered, local_credentials_rendered):
        assert 'application_default_credentials.json' not in rendered
        assert 'credentials.db' not in rendered
        assert 'auth/impersonate_service_account' not in rendered
        assert rendered.count('"~/.ssh/sky-cluster-key"') == 1


# --- Tests for _is_reservation_bound ---


class TestIsReservationBound:
    """Tests for DENSE/CALENDAR reservation detection."""

    @patch('sky.provision.gcp.instance_utils'
           '.GCPComputeInstance.load_resource')
    def test_dense_reservation_returns_true(self, mock_load):
        """DENSE reservation should trigger RESERVATION_BOUND."""
        from sky.provision.gcp.instance_utils import _is_reservation_bound
        mock_compute = MagicMock()
        mock_load.return_value = mock_compute
        mock_compute.reservations.return_value.list.return_value \
            .execute.return_value = {
                'items': [{
                    'name': 'my-dense-reservation',
                    'deploymentType': 'DENSE',
                }]
            }

        result = _is_reservation_bound('my-project', 'us-central1-a',
                                       'my-dense-reservation')
        assert result is True

        mock_compute.reservations.return_value.list.assert_called_once_with(
            project='my-project',
            zone='us-central1-a',
            filter='name=my-dense-reservation',
        )

    @patch('sky.provision.gcp.instance_utils'
           '.GCPComputeInstance.load_resource')
    def test_standard_reservation_returns_false(self, mock_load):
        """Standard (SPECIFIC) reservation should not trigger override."""
        from sky.provision.gcp.instance_utils import _is_reservation_bound
        mock_compute = MagicMock()
        mock_load.return_value = mock_compute
        mock_compute.reservations.return_value.list.return_value \
            .execute.return_value = {
                'items': [{
                    'name': 'my-standard-reservation',
                    'deploymentType': 'DEPLOYMENT_TYPE_UNSPECIFIED',
                }]
            }

        result = _is_reservation_bound('my-project', 'us-central1-a',
                                       'my-standard-reservation')
        assert result is False

    @patch('sky.provision.gcp.instance_utils'
           '.GCPComputeInstance.load_resource')
    def test_no_deployment_type_returns_false(self, mock_load):
        """Reservation without deploymentType field should not override."""
        from sky.provision.gcp.instance_utils import _is_reservation_bound
        mock_compute = MagicMock()
        mock_load.return_value = mock_compute
        mock_compute.reservations.return_value.list.return_value \
            .execute.return_value = {
                'items': [{
                    'name': 'my-reservation',
                }]
            }

        result = _is_reservation_bound('my-project', 'us-central1-a',
                                       'my-reservation')
        assert result is False

    @patch('sky.provision.gcp.instance_utils'
           '.GCPComputeInstance.load_resource')
    def test_reservation_not_found_returns_false(self, mock_load):
        """Empty list result should return False."""
        from sky.provision.gcp.instance_utils import _is_reservation_bound
        mock_compute = MagicMock()
        mock_load.return_value = mock_compute
        mock_compute.reservations.return_value.list.return_value \
            .execute.return_value = {
                'items': []
            }

        result = _is_reservation_bound('my-project', 'us-central1-a',
                                       'nonexistent-reservation')
        assert result is False

    @patch('sky.provision.gcp.instance_utils'
           '.GCPComputeInstance.load_resource')
    def test_api_failure_returns_false(self, mock_load):
        """API errors should gracefully fall back to False."""
        from sky.provision.gcp.instance_utils import _is_reservation_bound
        mock_compute = MagicMock()
        mock_load.return_value = mock_compute
        mock_compute.reservations.return_value.list.return_value \
            .execute.side_effect = Exception('Permission denied')

        result = _is_reservation_bound('my-project', 'us-central1-a',
                                       'my-reservation')
        assert result is False

    @patch('sky.provision.gcp.instance_utils'
           '.GCPComputeInstance.load_resource')
    def test_full_uri_parses_short_name(self, mock_load):
        """Full reservation URI should be parsed to short name for API call."""
        from sky.provision.gcp.instance_utils import _is_reservation_bound
        mock_compute = MagicMock()
        mock_load.return_value = mock_compute
        mock_compute.reservations.return_value.list.return_value \
            .execute.return_value = {
                'items': [{
                    'name': 'my-dense-res',
                    'deploymentType': 'DENSE',
                }]
            }

        full_uri = ('projects/my-project/zones/us-central1-a'
                    '/reservations/my-dense-res')
        result = _is_reservation_bound('my-project', 'us-central1-a', full_uri)
        assert result is True

        mock_compute.reservations.return_value.list.assert_called_once_with(
            project='my-project',
            zone='us-central1-a',
            filter='name=my-dense-res',
        )
