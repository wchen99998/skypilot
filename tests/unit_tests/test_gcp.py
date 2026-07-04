import pathlib
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from sky import logs
from sky import resources
from sky import skypilot_config
from sky.backends import backend_utils
from sky.catalog import gcp_catalog
from sky.clouds import gcp as gcp_cloud
from sky.clouds import Region
from sky.clouds import Zone
from sky.clouds.gcp import GCP
from sky.clouds.utils import gcp_utils
from sky.provision import common
from sky.provision.gcp import config as gcp_config
from sky.provision.gcp import constants as gcp_constants
from sky.provision.gcp import instance_utils
from sky.provision.gcp import mig_utils
from sky.utils import common_utils
from sky.utils import config_utils
from sky.utils import resources_utils


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


def test_gcp_ct6e_machine_type_uses_mig_without_guest_accelerators():
    assert instance_utils.get_node_type({
        gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG: {
            'run_duration': 3600,
            'provision_timeout': 900,
        },
        'machineType': 'ct6e-standard-8t',
    }) == instance_utils.GCPNodeType.MIG


def test_gcp_cpu_machine_type_with_mig_config_uses_compute():
    assert instance_utils.get_node_type({
        gcp_constants.MANAGED_INSTANCE_GROUP_CONFIG: {
            'run_duration': 3600,
            'provision_timeout': 900,
        },
        'machineType': 'n1-standard-4',
    }) == instance_utils.GCPNodeType.COMPUTE


def test_gcp_ct6e_boot_disk_uses_hyperdisk_balanced():
    for disk_tier in (
            None,
            resources_utils.DiskTier.LOW,
            resources_utils.DiskTier.MEDIUM,
            resources_utils.DiskTier.HIGH,
            resources_utils.DiskTier.ULTRA,
            resources_utils.DiskTier.BEST,
    ):
        assert GCP._get_disk_type('ct6e-standard-4t',
                                  disk_tier) == 'hyperdisk-balanced'


def test_gcp_mig_resource_names_fit_gce_limit():
    cluster_name = 'spectra-' + 'long-' * 20

    for resource_name in (
            mig_utils.get_instance_template_name(cluster_name),
            mig_utils.get_managed_instance_group_name(cluster_name),
            mig_utils.get_workload_policy_name(cluster_name),
    ):
        assert len(resource_name) <= mig_utils.GCE_RESOURCE_NAME_MAX_LENGTH
        assert resource_name.startswith('sky-')


def test_gcp_tpu_mig_gcloud_commands(monkeypatch):
    commands = []

    class Result:
        stdout = ''
        stderr = ''
        returncode = 0

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return Result()

    monkeypatch.setattr(mig_utils.subprocess, 'run', fake_run)

    mig_utils.create_workload_policy('project', 'us-east5', 'policy', '4x8',
                                     'AUTO_CONNECT')
    mig_utils.create_region_managed_instance_group(
        'project',
        'us-east5',
        ['us-east5-a'],
        'mig',
        'projects/project/regions/us-east5/instanceTemplates/template',
        8,
        'projects/project/regions/us-east5/resourcePolicies/policy',
    )

    assert commands[0][0] == [
        'gcloud',
        '--project',
        'project',
        'compute',
        'resource-policies',
        'create',
        'workload-policy',
        'policy',
        '--type',
        'HIGH_THROUGHPUT',
        '--accelerator-topology',
        '4x8',
        '--accelerator-topology-mode',
        'AUTO_CONNECT',
        '--region',
        'us-east5',
        '--quiet',
    ]
    assert commands[1][0] == [
        'gcloud',
        '--project',
        'project',
        'compute',
        'instance-groups',
        'managed',
        'create',
        'mig',
        '--size',
        '8',
        '--target-size-policy-mode',
        'bulk',
        '--template',
        'projects/project/regions/us-east5/instanceTemplates/template',
        '--region',
        'us-east5',
        '--zones',
        'us-east5-a',
        '--default-action-on-vm-failure',
        'do-nothing',
        '--workload-policy',
        'projects/project/regions/us-east5/resourcePolicies/policy',
        '--target-distribution-shape',
        'any-single-zone',
        '--instance-redistribution-type',
        'none',
        '--quiet',
    ]
    for _, kwargs in commands:
        assert kwargs['stdout'] == mig_utils.subprocess.PIPE
        assert kwargs['stderr'] == mig_utils.subprocess.PIPE
        assert kwargs['text'] is True
        assert kwargs['check'] is False


def test_gcp_ct6e_mig_uses_regional_bulk_workload_policy(monkeypatch):
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
    monkeypatch.setattr(
        instance_utils.GCPManagedInstanceGroup, '_add_labels_and_find_head',
        classmethod(lambda cls, *args, **kwargs: ['node-0', 'node-1']))
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
                'accelerator_topology': '4x8',
                'accelerator_topology_mode': 'AUTO_CONNECT',
            },
            'machineType': 'ct6e-standard-4t',
            'labels': {},
        },
        labels={},
        count=2,
        total_count=2,
        include_head_node=True,
    )

    assert instance_names == ['node-0', 'node-1']
    assert [call[0] for call in calls] == [
        'template',
        'wait-operation',
        'workload-policy',
        'regional-mig',
        'wait-regional-mig',
        'head-tag',
    ]
    regional_mig_call = calls[3]
    assert regional_mig_call[2] == {
        'size': 2,
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


def test_gcp_ct6e_catalog_entries_are_launchable_in_us_east5():
    assert gcp_catalog.instance_type_exists('ct6e-standard-8t') is True
    assert gcp_catalog.get_vcpus_mem_from_instance_type('ct6e-standard-8t') == (
        360, 1440)

    regions = gcp_catalog.get_region_zones_for_instance_type('ct6e-standard-8t',
                                                             use_spot=False)

    assert [region.name for region in regions] == ['us-east5']
    assert [zone.name for zone in regions[0].zones] == [
        'us-east5-a',
        'us-east5-b',
        'us-east5-c',
    ]


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
    assert calls == [('tpu-v6e-8', 1, False, 'us-east5', 'us-east5-a')]


def test_gcp_ct6e_spot_is_not_launchable():
    assert gcp_catalog.get_region_zones_for_instance_type('ct6e-standard-8t',
                                                          use_spot=True) == []
    with pytest.raises(ValueError, match='do not support spot'):
        gcp_catalog.get_hourly_cost('ct6e-standard-8t', use_spot=True)


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
