"""Managed Instance Group Utils"""
import hashlib
import re
import time
from typing import Any, Dict, List

from sky import sky_logging
from sky.adaptors import gcp
from sky.provision.gcp import constants

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


def check_instance_template_exits(project_id: str, region: str,
                                  template_name: str) -> bool:
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    try:
        compute.regionInstanceTemplates().get(
            project=project_id, region=region,
            instanceTemplate=template_name).execute(
                num_retries=_GCP_API_MAX_RETRIES)
    except gcp.http_error_exception() as e:
        if IT_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is not None:
            # Instance template does not exist.
            return False
        raise
    return True


def create_region_instance_template(cluster_name_on_cloud: str, project_id: str,
                                    region: str, template_name: str,
                                    node_config: Dict[str, Any]) -> dict:
    """Create a regional instance template."""
    logger.debug(f'Creating regional instance template {template_name!r}.')
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    config = node_config.copy()
    managed_instance_group_config = config.pop(
        constants.MANAGED_INSTANCE_GROUP_CONFIG, None)
    assert managed_instance_group_config is not None, (
        'Managed instance group config is required for DWS.')

    scheduling = config.get('scheduling', {})
    assert scheduling.get('provisioningModel') != 'SPOT', (
        'DWS does not support spot VMs.')
    if scheduling:
        logger.warning(
            f'Ignoring scheduling {scheduling} for DWS. DWS requires '
            'Flex-start scheduling.')
    config['scheduling'] = {
        'provisioningModel': 'FLEX_START',
        'instanceTerminationAction': 'DELETE',
        'maxRunDuration': {
            'seconds': str(managed_instance_group_config['run_duration']),
        },
        'onHostMaintenance': 'TERMINATE',
    }

    reservations_affinity = config.pop('reservationAffinity', None)
    legacy_reservations_affinity = config.pop('reservation_affinity', None)
    if reservations_affinity is None:
        reservations_affinity = legacy_reservations_affinity
    if reservations_affinity is not None:
        logger.warning(
            f'Ignoring reservations_affinity {reservations_affinity} '
            'for DWS.')

    config['description'] = ('SkyPilot instance template for '
                             f'{cluster_name_on_cloud!r} to support DWS '
                             'requests.')
    config['reservationAffinity'] = {
        'consumeReservationType': 'NO_RESERVATION',
    }

    # Create the regional instance template request
    operation = compute.regionInstanceTemplates().insert(
        project=project_id,
        region=region,
        body={
            'name': template_name,
            'properties': config,
        }).execute(num_retries=_GCP_API_MAX_RETRIES)
    return operation


def check_workload_policy_exists(project_id: str, region: str,
                                 policy_name: str) -> bool:
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    try:
        compute.resourcePolicies().get(project=project_id,
                                       region=region,
                                       resourcePolicy=policy_name).execute(
                                           num_retries=_GCP_API_MAX_RETRIES)
    except gcp.http_error_exception() as e:
        if WORKLOAD_POLICY_RESOURCE_NOT_FOUND_PATTERN.search(
                str(e)) is not None:
            return False
        raise
    return True


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
        body={
            'name': policy_name,
            'workloadPolicy': {
                'type': TPU_MIG_WORKLOAD_POLICY_TYPE,
                'acceleratorTopology': accelerator_topology,
                'acceleratorTopologyMode': accelerator_topology_mode,
            },
        }).execute(num_retries=_GCP_API_MAX_RETRIES)


def delete_workload_policy(project_id: str, region: str,
                           policy_name: str) -> None:
    logger.debug(f'Deleting workload policy {policy_name!r}.')
    try:
        compute = gcp.build('compute',
                            'v1',
                            credentials=None,
                            cache_discovery=False)
        operation = compute.resourcePolicies().delete(
            project=project_id, region=region,
            resourcePolicy=policy_name).execute(
                num_retries=_GCP_API_MAX_RETRIES)
        compute.regionOperations().wait(project=project_id,
                                        region=region,
                                        operation=operation['name']).execute(
                                            num_retries=_GCP_API_MAX_RETRIES)
    except gcp.http_error_exception() as e:
        if WORKLOAD_POLICY_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is None:
            raise
        logger.debug(f'Workload policy {policy_name!r} does not exist. Skip '
                     'deletion.')


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
        project=project_id, region=region,
        body=body).execute(num_retries=_GCP_API_MAX_RETRIES)


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


def check_region_managed_instance_group_exists(project_id: str, region: str,
                                               group_name: str) -> bool:
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    try:
        compute.regionInstanceGroupManagers().get(
            project=project_id, region=region,
            instanceGroupManager=group_name).execute(
                num_retries=_GCP_API_MAX_RETRIES)
    except gcp.http_error_exception() as e:
        if REGION_MIG_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is not None:
            return False
        raise
    return True


def list_managed_instance_group_instances(project_id: str, zone: str,
                                          group_name: str) -> List[str]:
    """Return the instances owned by a regional or zonal MIG.

    MIG membership is the authoritative cluster boundary. Labels can be
    temporarily stale or incorrectly copied from another instance template,
    so status queries must not use labels to decide which VMs belong to a MIG.
    """
    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)

    def list_instances(managers: Any, **kwargs: str) -> List[str]:
        instance_names = []
        while True:
            response = managers.listManagedInstances(**kwargs).execute(
                num_retries=_GCP_API_MAX_RETRIES)
            instance_names.extend(
                instance['name']
                for instance in response.get('managedInstances', []))
            page_token = response.get('nextPageToken')
            if page_token is None:
                return instance_names
            kwargs['pageToken'] = page_token

    region = zone.rpartition('-')[0]
    try:
        return list_instances(
            compute.regionInstanceGroupManagers(),
            project=project_id,
            region=region,
            instanceGroupManager=group_name,
        )
    except gcp.http_error_exception() as e:
        if REGION_MIG_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is None:
            raise

    try:
        return list_instances(
            compute.instanceGroupManagers(),
            project=project_id,
            zone=zone,
            instanceGroupManager=group_name,
        )
    except gcp.http_error_exception() as e:
        if MIG_RESOURCE_NOT_FOUND_PATTERN.search(str(e)) is None:
            raise
        return []


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

    compute = gcp.build('compute',
                        'v1',
                        credentials=None,
                        cache_discovery=False)
    if regional:
        managers = compute.regionInstanceGroupManagers()
        get_kwargs = {
            'project': project_id,
            'region': location,
            'instanceGroupManager': group_name,
        }
    else:
        managers = compute.instanceGroupManagers()
        get_kwargs = {
            'project': project_id,
            'zone': location,
            'instanceGroupManager': group_name,
        }

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
    while True:
        last_result = managers.get(**get_kwargs).execute(
            num_retries=_GCP_API_MAX_RETRIES)
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
