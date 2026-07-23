"""Managed Instance Group Utils"""
import copy
import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional
import uuid

from sky import sky_logging
from sky.adaptors import gcp
from sky.provision.gcp import constants
from sky.utils import annotations

logger = sky_logging.init_logger(__name__)

MIG_RESOURCE_NOT_FOUND_PATTERN = re.compile(
    r'The resource \'projects/.*/zones/.*/instanceGroupManagers/.*\' was not '
    r'found')

REGION_MIG_RESOURCE_NOT_FOUND_PATTERN = re.compile(
    r'The resource \'projects/.*/regions/.*/instanceGroupManagers/.*\' was '
    r'not found')

IT_RESOURCE_NOT_FOUND_PATTERN = re.compile(
    r'The resource \'projects/.*/regions/.*/instanceTemplates/.*\' was not '
    'found')

WORKLOAD_POLICY_RESOURCE_NOT_FOUND_PATTERN = re.compile(
    r'The resource \'projects/.*/regions/.*/resourcePolicies/.*\' was not '
    'found')

GCE_RESOURCE_NAME_MAX_LENGTH = 63
CT6E_MACHINE_TYPE_PREFIX = 'ct6e-standard-'
TPU7X_MACHINE_TYPE_PREFIX = 'tpu7x-standard-'
COMPUTE_TPU_MACHINE_TYPE_PREFIXES = (CT6E_MACHINE_TYPE_PREFIX,
                                     TPU7X_MACHINE_TYPE_PREFIX)
TPU_MIG_DEFAULT_ACCELERATOR_TOPOLOGY_MODE = 'AUTO_CONNECT'
TPU_MIG_WORKLOAD_POLICY_TYPE = 'HIGH_THROUGHPUT'
_TPU_FLEX_START_ZONES = {
    'ct6e': {
        'asia-northeast1-b',
        'us-east5-a',
        'us-south1-ai1b',
    },
    'tpu7x': {'us-central1-c',},
}
# Compute Engine supports a discrete set of TPU slice shapes. In particular,
# v6e 2x4 on two ct6e-standard-4t VMs is GKE-only and must not be accepted by
# this Compute Engine provisioner.
_COMPUTE_TPU_TOPOLOGY_TO_VM_COUNT = {
    'ct6e-standard-1t': {
        '1x1': 1,
    },
    'ct6e-standard-4t': {
        '2x2': 1,
        '4x4': 4,
        '4x8': 8,
        '8x8': 16,
        '8x16': 32,
        '16x16': 64,
    },
    'ct6e-standard-8t': {
        '2x4': 1,
    },
    'tpu7x-standard-4t': {
        '2x2x1': 1,
        '2x2x2': 2,
        '2x2x4': 4,
        '2x4x4': 8,
        '4x4x4': 16,
        '4x4x8': 32,
        '4x8x8': 64,
        '8x8x8': 128,
        '8x8x16': 256,
        '8x16x16': 512,
    },
}
_TPU_TOPOLOGY_PATTERN = re.compile(r'^[1-9][0-9]*(?:x[1-9][0-9]*)+$')
_GCP_API_MAX_RETRIES = 5
_BULK_MIG_POLL_INTERVAL_SECONDS = 10
_GCP_AUTH_REFRESH_RETRY_INTERVAL_SECONDS = 10
_GCP_AUTH_REFRESH_WARNING_INTERVAL_SECONDS = 60
_GCP_TRANSIENT_RETRY_INTERVAL_SECONDS = 2
_GCP_TRANSIENT_HTTP_STATUS_CODES = frozenset({
    408,
    429,
    500,
    502,
    503,
    504,
})
_TEMPLATE_OUTPUT_ONLY_FIELDS = frozenset({'fingerprint'})
_TEMPLATE_UNORDERED_LIST_FIELDS = frozenset({
    'guestAccelerators',
    'items',
    'resourcePolicies',
    'scopes',
    'serviceAccounts',
})
_TEMPLATE_BASENAME_REFERENCE_FIELDS = frozenset({
    'acceleratorType',
    'diskType',
    'machineType',
})
_TEMPLATE_PATH_REFERENCE_FIELDS = frozenset({
    'network',
    'source',
    'sourceImage',
    'subnetwork',
})
_TEMPLATE_INTEGER_FIELDS = frozenset({
    'acceleratorCount',
    'diskSizeGb',
    'provisionedIops',
    'provisionedThroughput',
    'threadsPerCore',
    'visibleCoreCount',
})
_TEMPLATE_API_DEFAULTS = {
    'canIpForward': False,
    'deletionProtection': False,
    'interface': 'SCSI',
    'keyRevocationActionType': 'NONE',
    'nicType': 'VIRTIO_NET',
    'privateIpv6GoogleAccess': 'INHERIT_FROM_SUBNETWORK',
    'stackType': 'IPV4_ONLY',
}
_TEMPLATE_GUARDED_PROPERTIES = frozenset({
    'advancedMachineFeatures',
    'canIpForward',
    'confidentialInstanceConfig',
    'deletionProtection',
    'disks',
    'displayDevice',
    'guestAccelerators',
    'keyRevocationActionType',
    'labels',
    'metadata',
    'minCpuPlatform',
    'networkInterfaces',
    'networkPerformanceConfig',
    'params',
    'privateIpv6GoogleAccess',
    'resourceManagerTags',
    'resourcePolicies',
    'serviceAccounts',
    'shieldedInstanceConfig',
    'sourceMachineImage',
    'tags',
})
_MISSING = object()


@annotations.ttl_cache(scope='request', maxsize=32, ttl=60)
def get_missing_tpu_flex_start_permissions(project_id: str) -> List[str]:
    """Return missing project permissions for a regional TPU Flex-start MIG."""
    crm = gcp.build('cloudresourcemanager',
                    'v1',
                    credentials=None,
                    cache_discovery=False)
    requested_permissions = constants.TPU_FLEX_START_MIG_PERMISSIONS
    response = crm.projects().testIamPermissions(
        resource=project_id, body={
            'permissions': requested_permissions,
        }).execute(num_retries=_GCP_API_MAX_RETRIES)
    granted_permissions = set(response.get('permissions', []))
    return sorted(set(requested_permissions) - granted_permissions)


def _prefixed_resource_name(prefix: str, cluster_name: str) -> str:
    name = f'{prefix}{cluster_name}'
    if len(name) <= GCE_RESOURCE_NAME_MAX_LENGTH:
        return name
    digest = hashlib.sha1(cluster_name.encode()).hexdigest()[:8]
    base_length = GCE_RESOURCE_NAME_MAX_LENGTH - len(prefix) - len(digest) - 1
    base = cluster_name[:base_length].rstrip('-')
    return f'{prefix}{base}-{digest}'


def get_instance_template_name(cluster_name: str) -> str:
    return _prefixed_resource_name(constants.INSTANCE_TEMPLATE_NAME_PREFIX,
                                   cluster_name)


def get_managed_instance_group_name(cluster_name: str) -> str:
    return _prefixed_resource_name(constants.MIG_NAME_PREFIX, cluster_name)


def get_workload_policy_name(cluster_name: str) -> str:
    return _prefixed_resource_name(constants.WORKLOAD_POLICY_NAME_PREFIX,
                                   cluster_name)


def get_workload_policy_url(project_id: str, region: str,
                            policy_name: str) -> str:
    return (f'projects/{project_id}/regions/{region}/resourcePolicies/'
            f'{policy_name}')


def is_ct6e_machine_type(machine_type: str) -> bool:
    return machine_type.startswith(CT6E_MACHINE_TYPE_PREFIX)


def is_tpu7x_machine_type(machine_type: str) -> bool:
    return machine_type.startswith(TPU7X_MACHINE_TYPE_PREFIX)


def is_compute_tpu_machine_type(machine_type: str) -> bool:
    return machine_type.startswith(COMPUTE_TPU_MACHINE_TYPE_PREFIXES)


def validate_tpu_flex_start_config(machine_type: str, zone: str,
                                   accelerator_topology: str,
                                   total_count: int) -> None:
    """Validate a Compute Engine TPU Flex-start slice configuration."""
    if is_ct6e_machine_type(machine_type):
        tpu_series = 'ct6e'
        topology_dimensions = 2
    elif is_tpu7x_machine_type(machine_type):
        tpu_series = 'tpu7x'
        topology_dimensions = 3
    else:
        raise ValueError(
            f'Unsupported Compute Engine TPU machine type {machine_type!r}.')

    flex_start_zones = _TPU_FLEX_START_ZONES[tpu_series]
    if zone not in flex_start_zones:
        zones = ', '.join(sorted(flex_start_zones))
        raise ValueError(
            f'{machine_type!r} does not support TPU Flex-start in {zone!r}. '
            f'Use one of: {zones}.')

    if not _TPU_TOPOLOGY_PATTERN.fullmatch(accelerator_topology):
        raise ValueError('Invalid TPU accelerator topology '
                         f'{accelerator_topology!r}. Expected dimensions '
                         'separated by "x", such as "4x8" or "2x2x2".')
    dimensions = [int(value) for value in accelerator_topology.split('x')]
    if len(dimensions) != topology_dimensions:
        raise ValueError(
            f'{machine_type!r} requires a {topology_dimensions}D accelerator '
            f'topology; got {accelerator_topology!r}.')

    canonical_machine_type = (machine_type[:-4] if machine_type.endswith('-tpu')
                              else machine_type)
    topology_to_vm_count = _COMPUTE_TPU_TOPOLOGY_TO_VM_COUNT.get(
        canonical_machine_type)
    if topology_to_vm_count is None:
        raise ValueError(
            f'Unsupported Compute Engine TPU machine type {machine_type!r}.')
    expected_count = topology_to_vm_count.get(accelerator_topology)
    if expected_count is None:
        supported_topologies = ', '.join(topology_to_vm_count)
        raise ValueError(
            f'Accelerator topology {accelerator_topology!r} is not supported '
            f'with {machine_type!r} on Compute Engine. Supported topologies: '
            f'{supported_topologies}.')
    if expected_count != total_count:
        raise ValueError(
            f'Accelerator topology {accelerator_topology!r} requires '
            f'{expected_count} {machine_type!r} VMs, but num_nodes is '
            f'{total_count}.')


def is_tpu_managed_instance_group(node_config: Dict[str, Any]) -> bool:
    return (node_config.get(constants.MANAGED_INSTANCE_GROUP_CONFIG) is not None
            and is_compute_tpu_machine_type(node_config.get('machineType', '')))


def get_region_instance_template(project_id: str, region: str,
                                 template_name: str) -> Optional[dict]:
    """Return a regional instance template, or None if it does not exist."""
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    try:
        return compute.regionInstanceTemplates().get(
            project=project_id, region=region,
            instanceTemplate=template_name).execute(
                num_retries=_GCP_API_MAX_RETRIES)
    except gcp.http_error_exception() as e:
        if IT_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is not None:
            return None
        raise


def check_instance_template_exits(project_id: str, region: str,
                                  template_name: str) -> bool:
    return get_region_instance_template(project_id, region,
                                        template_name) is not None


def make_region_instance_template_properties(
        cluster_name_on_cloud: str, node_config: Dict[str,
                                                      Any]) -> Dict[str, Any]:
    """Build the intended properties for a TPU Flex-start template."""
    config = copy.deepcopy(node_config)
    managed_instance_group_config = config.pop(
        constants.MANAGED_INSTANCE_GROUP_CONFIG, None)
    assert managed_instance_group_config is not None, (
        'Managed instance group config is required for DWS.')

    scheduling = config.get('scheduling', {})
    assert scheduling.get('provisioningModel') != 'SPOT', (
        'DWS does not support spot VMs.')
    config['scheduling'] = {
        'provisioningModel': 'FLEX_START',
        'instanceTerminationAction': 'DELETE',
        'maxRunDuration': {
            'seconds': str(managed_instance_group_config['run_duration']),
        },
        'onHostMaintenance': 'TERMINATE',
    }

    config.pop('reservationAffinity', None)
    config.pop('reservation_affinity', None)
    config['description'] = ('SkyPilot instance template for '
                             f'{cluster_name_on_cloud!r} to support DWS '
                             'requests.')
    config['reservationAffinity'] = {
        'consumeReservationType': 'NO_RESERVATION',
    }
    return config


def create_region_instance_template(cluster_name_on_cloud: str, project_id: str,
                                    region: str, template_name: str,
                                    node_config: Dict[str, Any]) -> dict:
    """Create a regional instance template."""
    logger.debug(f'Creating regional instance template {template_name!r}.')
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    scheduling = node_config.get('scheduling', {})
    if scheduling:
        logger.warning(
            f'Ignoring scheduling {scheduling} for DWS. DWS requires '
            'Flex-start scheduling.')

    reservations_affinity = node_config.get('reservationAffinity')
    legacy_reservations_affinity = node_config.get('reservation_affinity')
    if reservations_affinity is None:
        reservations_affinity = legacy_reservations_affinity
    if reservations_affinity is not None:
        logger.warning(
            f'Ignoring reservations_affinity {reservations_affinity} '
            'for DWS.')
    properties = make_region_instance_template_properties(
        cluster_name_on_cloud, node_config)

    # Create the regional instance template request
    operation = compute.regionInstanceTemplates().insert(
        project=project_id,
        region=region,
        requestId=str(uuid.uuid4()),
        body={
            'name': template_name,
            'properties': properties,
        }).execute(num_retries=_GCP_API_MAX_RETRIES)
    return operation


def get_workload_policy(project_id: str, region: str,
                        policy_name: str) -> Optional[dict]:
    """Return a regional resource policy, or None if it does not exist."""
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    try:
        return compute.resourcePolicies().get(
            project=project_id, region=region,
            resourcePolicy=policy_name).execute(
                num_retries=_GCP_API_MAX_RETRIES)
    except gcp.http_error_exception() as e:
        if WORKLOAD_POLICY_RESOURCE_NOT_FOUND_PATTERN.search(
                str(e)) is not None:
            return None
        raise


def check_workload_policy_exists(project_id: str, region: str,
                                 policy_name: str) -> bool:
    return get_workload_policy(project_id, region, policy_name) is not None


def create_workload_policy(project_id: str, region: str, policy_name: str,
                           accelerator_topology: str,
                           accelerator_topology_mode: str) -> dict:
    logger.debug(f'Creating workload policy {policy_name!r}.')
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    return compute.resourcePolicies().insert(
        project=project_id,
        region=region,
        requestId=str(uuid.uuid4()),
        body={
            'name': policy_name,
            'workloadPolicy': {
                'type': TPU_MIG_WORKLOAD_POLICY_TYPE,
                'acceleratorTopology': accelerator_topology,
                'acceleratorTopologyMode': accelerator_topology_mode,
            },
        }).execute(num_retries=_GCP_API_MAX_RETRIES)


def delete_workload_policy(project_id: str, region: str,
                           policy_name: str) -> Optional[dict]:
    logger.debug(f'Deleting workload policy {policy_name!r}.')
    try:
        compute = gcp.build('compute',
                            'v1',
                            credentials=None,
                            cache_discovery=False)
        return compute.resourcePolicies().delete(
            project=project_id, region=region,
            resourcePolicy=policy_name).execute(
                num_retries=_GCP_API_MAX_RETRIES)
    except gcp.http_error_exception() as e:
        if WORKLOAD_POLICY_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is None:
            raise
        logger.debug(f'Workload policy {policy_name!r} does not exist. Skip '
                     'deletion.')
        return None


def create_region_managed_instance_group(project_id: str, region: str,
                                         zones: List[str], group_name: str,
                                         instance_template_url: str, size: int,
                                         workload_policy_url: str) -> dict:
    logger.debug(f'Creating regional managed instance group {group_name!r}.')
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    body = {
        'name': group_name,
        'instanceTemplate': instance_template_url,
        'targetSize': size,
        'targetSizePolicy': {
            'mode': 'BULK',
        },
        'distributionPolicy': {
            'targetShape': 'ANY_SINGLE_ZONE',
            'zones': [{
                'zone': f'projects/{project_id}/zones/{zone}',
            } for zone in zones],
        },
        'instanceLifecyclePolicy': {
            'defaultActionOnFailure': 'DO_NOTHING',
        },
        'resourcePolicies': {
            'workloadPolicy': workload_policy_url,
        },
        'updatePolicy': {
            'type': 'OPPORTUNISTIC',
            'instanceRedistributionType': 'NONE',
        },
    }
    return compute.regionInstanceGroupManagers().insert(
        project=project_id,
        region=region,
        requestId=str(uuid.uuid4()),
        body=body).execute(num_retries=_GCP_API_MAX_RETRIES)


def _resource_reference_matches(value: Any, expected_path: str) -> bool:
    if not isinstance(value, str):
        return False
    value = value.rstrip('/')
    expected_path = expected_path.rstrip('/')
    if 'projects/' in value:
        return value == expected_path or value.endswith(f'/{expected_path}')
    # The Compute API also accepts and can expose same-project partial URLs,
    # such as regions/<region>/resourcePolicies/<name>.
    expected_partial_path = expected_path.split('/', 2)[2]
    return (value == expected_partial_path or
            value.endswith(f'/{expected_partial_path}'))


def _duration_seconds_matches(value: Any, expected_seconds: int) -> bool:
    if not isinstance(value, dict):
        return False
    seconds = value.get('seconds')
    if not isinstance(seconds, (int, str)):
        return False
    try:
        return int(seconds) == expected_seconds
    except ValueError:
        return False


def _normalize_template_reference(field: Optional[str], value: str) -> str:
    if field in _TEMPLATE_BASENAME_REFERENCE_FIELDS:
        return value.rstrip('/').rpartition('/')[2]
    if field in _TEMPLATE_PATH_REFERENCE_FIELDS:
        match = re.search(r'/(projects/[^/]+/.+)$', value)
        if match is not None:
            return match.group(1)
    return value


def _is_default_template_value(field: str, value: Any) -> bool:
    return (field in _TEMPLATE_API_DEFAULTS and
            value == _TEMPLATE_API_DEFAULTS[field])


def _normalize_template_value(value: Any,
                              expected: Any = _MISSING,
                              field: Optional[str] = None) -> Any:
    """Normalize API-expanded instance-template properties for comparison."""
    if isinstance(value, dict):
        expected_dict = expected if isinstance(expected, dict) else {}
        normalized = {}
        for key, child in value.items():
            if key in _TEMPLATE_OUTPUT_ONLY_FIELDS:
                continue
            expected_child = expected_dict.get(key, _MISSING)
            if (expected_child is _MISSING and
                    _is_default_template_value(key, child)):
                continue
            normalized[key] = _normalize_template_value(child,
                                                        expected=expected_child,
                                                        field=key)
        return normalized
    if isinstance(value, list):
        expected_list = expected if isinstance(expected, list) else []
        normalized_list = [
            _normalize_template_value(
                item,
                expected=(expected_list[index]
                          if index < len(expected_list) else _MISSING),
                field=field) for index, item in enumerate(value)
        ]
        if field in _TEMPLATE_UNORDERED_LIST_FIELDS:
            normalized_list.sort(
                key=lambda item: json.dumps(item, sort_keys=True))
        return normalized_list
    if isinstance(value, str):
        if field in _TEMPLATE_INTEGER_FIELDS:
            try:
                return int(value)
            except ValueError:
                pass
        return _normalize_template_reference(field, value)
    return value


def _template_property_matches(actual: Any, expected: Any, field: str) -> bool:
    return (_normalize_template_value(actual, expected=expected,
                                      field=field) == _normalize_template_value(
                                          expected,
                                          expected=expected,
                                          field=field))


def get_tpu_mig_reuse_mismatches(
        project_id: str, region: str, zone: str, total_count: int,
        machine_type: str, run_duration: int, accelerator_topology: str,
        accelerator_topology_mode: str, instance_template_name: str,
        workload_policy_name: str, expected_instance_properties: Dict[str, Any],
        instance_template: Optional[dict], workload_policy: Optional[dict],
        managed_instance_group: Optional[dict],
        managed_instances: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Return reasons existing TPU Flex-start resources cannot be reused."""
    mismatches = []
    instance_template_path = (
        f'projects/{project_id}/regions/{region}/instanceTemplates/'
        f'{instance_template_name}')
    workload_policy_path = (
        f'projects/{project_id}/regions/{region}/resourcePolicies/'
        f'{workload_policy_name}')

    if instance_template is None:
        mismatches.append('regional instance template is missing')
    else:
        properties = instance_template.get('properties', {})
        if not isinstance(properties, dict):
            properties = {}
        actual_machine_type = properties.get('machineType')
        if (not isinstance(actual_machine_type, str) or
                actual_machine_type.rpartition('/')[2] !=
                machine_type.rpartition('/')[2]):
            mismatches.append(
                f'instance template machine type is {actual_machine_type!r}, '
                f'expected {machine_type!r}')
        scheduling = properties.get('scheduling', {})
        if not isinstance(scheduling, dict):
            scheduling = {}
        if scheduling.get('provisioningModel') != 'FLEX_START':
            mismatches.append('instance template is not FLEX_START')
        if not _duration_seconds_matches(scheduling.get('maxRunDuration'),
                                         run_duration):
            mismatches.append('instance template maxRunDuration does not match '
                              f'{run_duration} seconds')
        if scheduling.get('instanceTerminationAction') != 'DELETE':
            mismatches.append(
                'instance template termination action is not DELETE')
        if scheduling.get('onHostMaintenance') != 'TERMINATE':
            mismatches.append(
                'instance template host maintenance action is not TERMINATE')
        canonical_machine_type = (machine_type[:-4]
                                  if machine_type.endswith('-tpu') else
                                  machine_type)
        if canonical_machine_type == 'ct6e-standard-8t':
            advanced_machine_features = properties.get(
                'advancedMachineFeatures', {})
            if not isinstance(advanced_machine_features, dict):
                advanced_machine_features = {}
            if advanced_machine_features.get('threadsPerCore') != 1:
                mismatches.append(
                    'instance template threadsPerCore is not 1 as required '
                    'for ct6e-standard-8t')
        comparison_properties = (set(expected_instance_properties) |
                                 _TEMPLATE_GUARDED_PROPERTIES) - {
                                     'description',
                                     'scheduling',
                                 }
        for property_name in sorted(comparison_properties):
            expected_value = expected_instance_properties.get(
                property_name, _MISSING)
            actual_value = properties.get(property_name, _MISSING)
            if expected_value is _MISSING:
                if (actual_value is not _MISSING and
                        not _is_default_template_value(property_name,
                                                       actual_value) and
                        _normalize_template_value(actual_value,
                                                  field=property_name)
                        not in (None, {}, [])):
                    mismatches.append(
                        f'instance template property {property_name!r} is '
                        'present but was not requested')
            elif (actual_value is _MISSING or not _template_property_matches(
                    actual_value, expected_value, property_name)):
                mismatches.append(
                    f'instance template property {property_name!r} does not '
                    'match the requested configuration')

    if workload_policy is None:
        mismatches.append('workload policy is missing')
    else:
        policy = workload_policy.get('workloadPolicy', {})
        if not isinstance(policy, dict):
            policy = {}
        if policy.get('type') != TPU_MIG_WORKLOAD_POLICY_TYPE:
            mismatches.append(
                f'workload policy type is {policy.get("type")!r}, expected '
                f'{TPU_MIG_WORKLOAD_POLICY_TYPE!r}')
        if policy.get('acceleratorTopology') != accelerator_topology:
            mismatches.append(
                'workload policy accelerator topology is '
                f'{policy.get("acceleratorTopology")!r}, expected '
                f'{accelerator_topology!r}')
        if (policy.get('acceleratorTopologyMode') != accelerator_topology_mode):
            mismatches.append(
                'workload policy accelerator topology mode is '
                f'{policy.get("acceleratorTopologyMode")!r}, expected '
                f'{accelerator_topology_mode!r}')

    if managed_instance_group is None:
        mismatches.append('regional managed instance group is missing')
    else:
        if managed_instance_group.get('targetSize') != total_count:
            mismatches.append(
                'regional managed instance group target size is '
                f'{managed_instance_group.get("targetSize")!r}, expected '
                f'{total_count}')
        target_size_policy = managed_instance_group.get('targetSizePolicy', {})
        if not isinstance(target_size_policy, dict):
            target_size_policy = {}
        if target_size_policy.get('mode') != 'BULK':
            mismatches.append(
                'regional managed instance group target size mode is not BULK')
        distribution_policy = managed_instance_group.get(
            'distributionPolicy', {})
        if not isinstance(distribution_policy, dict):
            distribution_policy = {}
        if distribution_policy.get('targetShape') != 'ANY_SINGLE_ZONE':
            mismatches.append(
                'regional managed instance group distribution target shape '
                'is not ANY_SINGLE_ZONE')
        distribution_zones = distribution_policy.get('zones', [])
        actual_zones = []
        if isinstance(distribution_zones, list):
            for distribution_zone in distribution_zones:
                if isinstance(distribution_zone, dict):
                    zone_reference = distribution_zone.get('zone')
                    if isinstance(zone_reference, str):
                        actual_zones.append(zone_reference.rpartition('/')[2])
        if actual_zones != [zone]:
            mismatches.append(
                'regional managed instance group distribution zones are '
                f'{actual_zones!r}, expected {[zone]!r}')
        if not _resource_reference_matches(
                managed_instance_group.get('instanceTemplate'),
                instance_template_path):
            mismatches.append(
                'regional managed instance group does not reference the '
                'intended instance template')
        resource_policies = managed_instance_group.get('resourcePolicies', {})
        if not isinstance(resource_policies, dict):
            resource_policies = {}
        if not _resource_reference_matches(
                resource_policies.get('workloadPolicy'), workload_policy_path):
            mismatches.append(
                'regional managed instance group does not reference the '
                'intended workload policy')
        update_policy = managed_instance_group.get('updatePolicy', {})
        if not isinstance(update_policy, dict):
            update_policy = {}
        if update_policy.get('type') != 'OPPORTUNISTIC':
            mismatches.append(
                'regional managed instance group update type is not '
                'OPPORTUNISTIC')
        if update_policy.get('instanceRedistributionType') != 'NONE':
            mismatches.append(
                'regional managed instance group instance redistribution is '
                'not disabled')
        instance_lifecycle_policy = managed_instance_group.get(
            'instanceLifecyclePolicy', {})
        if not isinstance(instance_lifecycle_policy, dict):
            instance_lifecycle_policy = {}
        if (instance_lifecycle_policy.get('defaultActionOnFailure') !=
                'DO_NOTHING'):
            mismatches.append(
                'regional managed instance group default action on failure '
                'is not DO_NOTHING')

        all_instances_config = managed_instance_group.get(
            'allInstancesConfig', {})
        if not isinstance(all_instances_config, dict):
            mismatches.append('regional managed instance group has an invalid '
                              'all-instances configuration')
        else:
            all_instances_properties = all_instances_config.get(
                'properties', {})
            if (not isinstance(all_instances_properties, dict) or
                    all_instances_properties):
                mismatches.append(
                    'regional managed instance group has all-instances '
                    'properties that override the intended instance template')

        stateful_policy = managed_instance_group.get('statefulPolicy', {})
        if not isinstance(stateful_policy, dict) or stateful_policy:
            mismatches.append(
                'regional managed instance group has a stateful policy')
        status = managed_instance_group.get('status', {})
        if not isinstance(status, dict):
            status = {}
        if total_count > 0:
            version_target_status = status.get('versionTarget', {})
            if (not isinstance(version_target_status, dict) or
                    version_target_status.get('isReached') is not True):
                mismatches.append(
                    'regional managed instance group target version is not '
                    'fully applied')
            all_instances_config_status = status.get('allInstancesConfig', {})
            if (not isinstance(all_instances_config_status, dict) or
                    all_instances_config_status.get('effective') is not True):
                mismatches.append(
                    'regional managed instance group all-instances '
                    'configuration is not fully applied')
        stateful_status = status.get('stateful', {})
        if not isinstance(stateful_status, dict):
            mismatches.append(
                'regional managed instance group has invalid stateful status')
        else:
            if stateful_status.get('hasStatefulConfig', False):
                mismatches.append(
                    'regional managed instance group has stateful '
                    'configuration')
            if total_count > 0:
                per_instance_config_status = stateful_status.get(
                    'perInstanceConfigs', {})
                if (not isinstance(per_instance_config_status, dict) or
                        per_instance_config_status.get('allEffective')
                        is not True):
                    mismatches.append(
                        'regional managed instance group per-instance '
                        'configurations are not fully applied')

        assert managed_instances is not None
        for managed_instance in managed_instances:
            instance_url = managed_instance.get('instance')
            instance_name = (instance_url.rpartition('/')[2]
                             if isinstance(instance_url, str) and instance_url
                             else '<pending instance>')
            version = managed_instance.get('version', {})
            if not isinstance(version, dict) or not _resource_reference_matches(
                    version.get('instanceTemplate'), instance_template_path):
                mismatches.append(
                    f'managed instance {instance_name!r} does not use the '
                    'intended instance template')
            for preserved_state_field in ('preservedStateFromConfig',
                                          'preservedStateFromPolicy'):
                preserved_state = managed_instance.get(preserved_state_field,
                                                       {})
                if (not isinstance(preserved_state, dict) or preserved_state):
                    mismatches.append(
                        f'managed instance {instance_name!r} has stateful '
                        'configuration')
                    break

    return mismatches


def create_managed_instance_group(project_id: str, zone: str, group_name: str,
                                  instance_template_url: str,
                                  size: int) -> dict:
    logger.debug(f'Creating managed instance group {group_name!r}.')
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    operation = compute.instanceGroupManagers().insert(
        project=project_id,
        zone=zone,
        body={
            'name': group_name,
            'instanceTemplate': instance_template_url,
            'target_size': size,
            'instanceLifecyclePolicy': {
                'defaultActionOnFailure': 'DO_NOTHING',
            },
            'updatePolicy': {
                'type': 'OPPORTUNISTIC',
            },
        }).execute()
    return operation


def resize_managed_instance_group(project_id: str, zone: str, group_name: str,
                                  resize_by: int, run_duration: int) -> dict:
    logger.debug(f'Resizing managed instance group {group_name!r} by '
                 f'{resize_by} with run duration {run_duration}.')
    compute = gcp.build('compute',
                        'beta',
                        credentials=None,
                        cache_discovery=False)
    operation = compute.instanceGroupManagerResizeRequests().insert(
        project=project_id,
        zone=zone,
        instanceGroupManager=group_name,
        body={
            'name': group_name,
            'resizeBy': resize_by,
            'requestedRunDuration': {
                'seconds': str(run_duration),
            }
        }).execute()
    return operation


def cancel_all_resize_request_for_mig(project_id: str, zone: str,
                                      group_name: str) -> None:
    logger.debug(f'Cancelling all resize requests for MIG {group_name!r}.')
    try:
        compute = gcp.build('compute',
                            'beta',
                            credentials=None,
                            cache_discovery=False)
        operation = compute.instanceGroupManagerResizeRequests().list(
            project=project_id,
            zone=zone,
            instanceGroupManager=group_name,
            filter='state eq ACCEPTED').execute()
        for request in operation.get('items', []):
            try:
                compute.instanceGroupManagerResizeRequests().cancel(
                    project=project_id,
                    zone=zone,
                    instanceGroupManager=group_name,
                    resizeRequest=request['name']).execute()
            except gcp.http_error_exception() as e:
                logger.warning('Failed to cancel resize request '
                               f'{request["id"]!r}: {e}')
    except gcp.http_error_exception() as e:
        if re.search(MIG_RESOURCE_NOT_FOUND_PATTERN, str(e)) is None:
            raise
        logger.warning(f'MIG {group_name!r} does not exist. Skip '
                       'resize request cancellation.')
        logger.debug(f'Error: {e}')


def check_managed_instance_group_exists(project_id: str, zone: str,
                                        group_name: str) -> bool:
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    try:
        compute.instanceGroupManagers().get(
            project=project_id, zone=zone,
            instanceGroupManager=group_name).execute()
    except gcp.http_error_exception() as e:
        if MIG_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is not None:
            return False
        raise
    return True


def get_region_managed_instance_group(project_id: str, region: str,
                                      group_name: str) -> Optional[dict]:
    """Return a regional managed instance group, or None if absent."""
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    try:
        return compute.regionInstanceGroupManagers().get(
            project=project_id, region=region,
            instanceGroupManager=group_name).execute(
                num_retries=_GCP_API_MAX_RETRIES)
    except gcp.http_error_exception() as e:
        if REGION_MIG_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is not None:
            return None
        raise


def check_region_managed_instance_group_exists(project_id: str, region: str,
                                               group_name: str) -> bool:
    return get_region_managed_instance_group(project_id, region,
                                             group_name) is not None


def _list_managed_instance_group_members(
        project_id: str, zone: str, group_name: str) -> List[Dict[str, Any]]:
    """Return the full managed-member records for a regional or zonal MIG."""
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)

    def list_members(managers: Any, **kwargs: str) -> List[Dict[str, Any]]:
        managed_instances = []
        while True:
            response = managers.listManagedInstances(**kwargs).execute(
                num_retries=_GCP_API_MAX_RETRIES)
            for managed_instance in response.get('managedInstances', []):
                if isinstance(managed_instance, dict):
                    managed_instances.append(managed_instance)
            page_token = response.get('nextPageToken')
            if page_token is None:
                return managed_instances
            kwargs['pageToken'] = page_token

    region = zone.rpartition('-')[0]
    try:
        return list_members(
            compute.regionInstanceGroupManagers(),
            project=project_id,
            region=region,
            instanceGroupManager=group_name,
        )
    except gcp.http_error_exception() as e:
        if REGION_MIG_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is None:
            raise

    try:
        return list_members(
            compute.instanceGroupManagers(),
            project=project_id,
            zone=zone,
            instanceGroupManager=group_name,
        )
    except gcp.http_error_exception() as e:
        if MIG_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is None:
            raise
        return []


def list_managed_instance_group_instances(project_id: str, zone: str,
                                          group_name: str) -> List[str]:
    """Return the instances owned by a regional or zonal MIG.

    MIG membership is the authoritative cluster boundary. Labels can be
    temporarily stale or incorrectly copied from another instance template,
    so status queries must not use labels to decide which VMs belong to a MIG.
    """
    instance_names = []
    for managed_instance in _list_managed_instance_group_members(
            project_id, zone, group_name):
        # ManagedInstance identifies its VM with the `instance` resource URL;
        # it has no `name` field. A target without a VM resource can appear
        # while a bulk request is still creating its instances.
        instance_url = managed_instance.get('instance')
        if not isinstance(instance_url, str) or not instance_url:
            continue
        instance_name = instance_url.rpartition('/')[2]
        if instance_name:
            instance_names.append(instance_name)
    return instance_names


def delete_region_managed_instance_group(project_id: str, region: str,
                                         group_name: str) -> dict:
    logger.debug(f'Deleting regional managed instance group {group_name!r}.')
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    operation = compute.regionInstanceGroupManagers().delete(
        project=project_id, region=region,
        instanceGroupManager=group_name).execute(
            num_retries=_GCP_API_MAX_RETRIES)
    return operation


def _wait_for_managed_group_to_be_stable(project_id: str, location: str,
                                         group_name: str, timeout: int,
                                         regional: bool) -> None:
    if timeout <= 0:
        raise ValueError(f'MIG provision timeout must be positive; got '
                         f'{timeout}.')

    if regional:
        get_kwargs = {
            'project': project_id,
            'region': location,
            'instanceGroupManager': group_name,
        }
    else:
        get_kwargs = {
            'project': project_id,
            'zone': location,
            'instanceGroupManager': group_name,
        }

    def _load_managers() -> Any:
        # Build with credentials=None so google-auth reloads Application Default
        # Credentials.  This matters when a long-running Flex-start request
        # outlives a user session and the ADC file is replaced after
        # reauthentication.
        compute = gcp.build('compute',
                            'v1',
                            credentials=None,
                            cache_discovery=False)
        if regional:
            return compute.regionInstanceGroupManagers()
        return compute.instanceGroupManagers()

    if regional:
        logger.info(
            'Waiting up to %s seconds for MIG %r to become stable. Google '
            'Cloud keeps this bulk TPU request queued and retries it until '
            'capacity is available.', timeout, group_name)
    else:
        logger.info('Waiting up to %s seconds for MIG %r to become stable.',
                    timeout, group_name)
    deadline = time.monotonic() + timeout
    last_progress = None
    last_result: Dict[str, Any] = {}
    managers = None
    last_auth_refresh_warning = None
    last_auth_error = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if last_auth_error is not None:
                raise TimeoutError(
                    f'Timed out after {timeout} seconds waiting for MIG '
                    f'{group_name!r} in {location!r} while reloading GCP '
                    'Application Default Credentials.') from last_auth_error
            status = last_result.get('status', {})
            progress_errors = (status.get('bulkInstanceOperation',
                                          {}).get('lastProgressCheck',
                                                  {}).get('error',
                                                          {}).get('errors', []))
            current_status = status.get('currentInstanceStatuses', {})
            raise TimeoutError(
                f'Timed out after {timeout} seconds waiting for MIG '
                f'{group_name!r} in {location!r}. Last progress errors: '
                f'{progress_errors}; instance status: {current_status}.')
        try:
            if managers is None:
                managers = _load_managers()
            request = managers.get(**get_kwargs)
            request_http = getattr(request, 'http', None)
            if request_http is not None and hasattr(request_http, 'timeout'):
                # The outer polling loop owns retries and the absolute
                # deadline. Bound this socket attempt by the remaining budget.
                request_http.timeout = max(0.1, remaining)
            last_result = request.execute(num_retries=0)
            last_auth_error = None
        except (gcp.gcp_auth_refresh_error_exception(),
                gcp.credential_error_exception()) as e:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                raise TimeoutError(
                    f'Timed out after {timeout} seconds waiting for MIG '
                    f'{group_name!r} in {location!r} while reloading GCP '
                    'Application Default Credentials.') from e
            last_auth_error = e
            if (last_auth_refresh_warning is None or
                    now - last_auth_refresh_warning >=
                    _GCP_AUTH_REFRESH_WARNING_INTERVAL_SECONDS):
                logger.warning(
                    'GCP credentials could not be refreshed while waiting for '
                    'MIG %r. Reloading Application Default Credentials until '
                    'the provision timeout (%d seconds remaining) without '
                    'resubmitting the MIG. Reauthenticate or rotate ADC in the '
                    'SkyPilot process environment.', group_name, int(remaining))
                last_auth_refresh_warning = now
            else:
                logger.debug(
                    'GCP credentials still cannot be refreshed for MIG %r; '
                    'reloading ADC again.', group_name)
            # Discard the discovery client and its in-memory Credentials
            # object.  A newly built client can observe an ADC file replaced
            # by `gcloud auth application-default login` or credential
            # rotation in the API-server environment.
            managers = None
            time.sleep(min(_GCP_AUTH_REFRESH_RETRY_INTERVAL_SECONDS, remaining))
            continue
        except gcp.http_error_exception() as e:
            status_code = getattr(getattr(e, 'resp', None), 'status', None)
            if status_code not in _GCP_TRANSIENT_HTTP_STATUS_CODES:
                raise
            last_auth_error = None
            managers = None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                continue
            logger.warning(
                'Transient GCP HTTP %s while waiting for MIG %r; retrying '
                'within the provision timeout.', status_code, group_name)
            time.sleep(min(_GCP_TRANSIENT_RETRY_INTERVAL_SECONDS, remaining))
            continue
        except (gcp.auth_transport_error_exception(),
                gcp.http_transport_error_exception(), OSError) as e:
            last_auth_error = None
            managers = None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                continue
            logger.warning(
                'Transient GCP transport error while waiting for MIG %r: %s. '
                'Retrying within the provision timeout.', group_name,
                type(e).__name__)
            time.sleep(min(_GCP_TRANSIENT_RETRY_INTERVAL_SECONDS, remaining))
            continue
        if time.monotonic() >= deadline:
            # Even a successful HTTP request must not extend the caller's
            # configured provision deadline.
            continue
        status = last_result.get('status', {})
        bulk_status = status.get('bulkInstanceOperation', {})
        if (status.get('isStable', False) and
                not bulk_status.get('inProgress', False)):
            instance_statuses = status.get('currentInstanceStatuses')
            if instance_statuses is None:
                if not regional:
                    return
                logger.debug(
                    'MIG %r reports stable but TPU instance statuses are not '
                    'available yet; waiting for the next status update.',
                    group_name)
            else:
                target_size = last_result.get('targetSize', 0)
                running = instance_statuses.get('running', 0)
                if running != target_size:
                    raise RuntimeError(
                        f'MIG {group_name!r} became stable with {running} of '
                        f'{target_size} requested instances running. Status: '
                        f'{instance_statuses}')
                return

        progress_check = bulk_status.get('lastProgressCheck', {})
        progress_errors = progress_check.get('error', {}).get('errors', [])
        progress = (
            progress_check.get('timestamp'),
            tuple((error.get('code'), error.get('message'))
                  for error in progress_errors),
        )
        if progress != last_progress and any(progress):
            if progress_errors:
                logger.info(
                    'MIG %r is still waiting. Latest Google Cloud '
                    'progress: %s', group_name, progress_errors)
            else:
                logger.debug('MIG %r progress checked at %s.', group_name,
                             progress_check.get('timestamp'))
            last_progress = progress

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            current_status = status.get('currentInstanceStatuses', {})
            raise TimeoutError(
                f'Timed out after {timeout} seconds waiting for MIG '
                f'{group_name!r} in {location!r}. Last progress errors: '
                f'{progress_errors}; instance status: {current_status}.')
        poll_interval = (_BULK_MIG_POLL_INTERVAL_SECONDS
                         if regional else constants.POLL_INTERVAL)
        time.sleep(min(poll_interval, remaining))


def wait_for_managed_group_to_be_stable(project_id: str, zone: str,
                                        group_name: str, timeout: int) -> None:
    """Wait until a zonal managed instance group is stable."""
    _wait_for_managed_group_to_be_stable(project_id,
                                         zone,
                                         group_name,
                                         timeout,
                                         regional=False)


def wait_for_region_managed_group_to_be_stable(project_id: str, region: str,
                                               group_name: str,
                                               timeout: int) -> None:
    """Wait until the regional managed instance group is stable."""
    _wait_for_managed_group_to_be_stable(project_id,
                                         region,
                                         group_name,
                                         timeout,
                                         regional=True)
