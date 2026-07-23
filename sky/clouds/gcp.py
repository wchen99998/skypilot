"""Google Cloud Platform."""
import configparser
import enum
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import typing
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import colorama

from sky import catalog
from sky import clouds
from sky import exceptions
from sky import sky_logging
from sky import skypilot_config
from sky.adaptors import gcp
from sky.clouds.utils import gcp_utils
from sky.clouds.utils import gpu_utils
from sky.provision.gcp import constants
from sky.provision.gcp import volume_utils
from sky.utils import annotations
from sky.utils import common_utils
from sky.utils import registry
from sky.utils import resources_utils
from sky.utils import subprocess_utils
from sky.utils import ux_utils

if typing.TYPE_CHECKING:
    from sky import resources
    from sky.utils import status_lib
    from sky.utils import volume as volume_lib

logger = sky_logging.init_logger(__name__)

# Env var pointing to any service account key. If it exists, this path takes
# priority over the DEFAULT_GCP_APPLICATION_CREDENTIAL_PATH below, and will be
# used instead for SkyPilot-launched instances. This is the same behavior as
# gcloud:
# https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key
_GCP_APPLICATION_CREDENTIAL_ENV = 'GOOGLE_APPLICATION_CREDENTIALS'
# NOTE: do not expanduser() on this path. It's used as a destination path on the
# remote cluster.
DEFAULT_GCP_APPLICATION_CREDENTIAL_PATH: str = (
    '~/.config/gcloud/'
    'application_default_credentials.json')

# TODO(wei-lin): config_default may not be the config in use.
# See: https://github.com/skypilot-org/skypilot/pull/1539
# NOTE: do not expanduser() on this path. It's used as a destination path on the
# remote cluster.
GCP_CONFIG_PATH = '~/.config/gcloud/configurations/config_default'

# Minimum set of files under ~/.config/gcloud that grant GCP access.
_GCLOUD_CONFIG_DIR = '~/.config/gcloud'
_STAGED_GCLOUD_CONFIG_DIR = '~/.sky/generated/gcp'

# NOTE: do not expanduser() on this path. It's used as a destination path on the
# remote cluster.
_GCLOUD_INSTALLATION_LOG = '~/.sky/logs/gcloud_installation.log'
# Bump carefully: this determines the gsutil version on remote VMs.
# 567.0.0 bundles gsutil 5.37 which fixes pyOpenSSL >= 24.3.0 compatibility
# (OpenSSL.crypto.sign was removed in pyOpenSSL 24.3.0).
# https://cloud.google.com/sdk/docs/release-notes#56700_2026-05-05
_GCLOUD_VERSION = '567.0.0'
# Need to be run with /bin/bash
# We factor out the installation logic to keep it align in both spot
# controller and cloud stores.
GOOGLE_SDK_INSTALLATION_COMMAND: str = f'pushd /tmp &>/dev/null && \
    {{ gcloud --help > /dev/null 2>&1 || \
    {{ mkdir -p {os.path.dirname(_GCLOUD_INSTALLATION_LOG)} && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        echo "Installing Google Cloud SDK for $ARCH" > {_GCLOUD_INSTALLATION_LOG} && \
        ARCH_SUFFIX="x86_64"; \
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        echo "Installing Google Cloud SDK for $ARCH" > {_GCLOUD_INSTALLATION_LOG} && \
        ARCH_SUFFIX="arm"; \
    else \
        echo "Architecture $ARCH not supported by Google Cloud SDK. Defaulting to x86_64." > {_GCLOUD_INSTALLATION_LOG} && \
        ARCH_SUFFIX="x86_64"; \
    fi && \
    echo "Detected architecture: $ARCH, using package: $ARCH_SUFFIX" >> {_GCLOUD_INSTALLATION_LOG} && \
    wget --quiet https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-{_GCLOUD_VERSION}-linux-${{ARCH_SUFFIX}}.tar.gz >> {_GCLOUD_INSTALLATION_LOG} && \
    tar xzf google-cloud-sdk-{_GCLOUD_VERSION}-linux-${{ARCH_SUFFIX}}.tar.gz >> {_GCLOUD_INSTALLATION_LOG} && \
    rm -rf ~/google-cloud-sdk >> {_GCLOUD_INSTALLATION_LOG}  && \
    mv google-cloud-sdk ~/ && \
    ~/google-cloud-sdk/install.sh -q >> {_GCLOUD_INSTALLATION_LOG} 2>&1 && \
    echo "source ~/google-cloud-sdk/path.bash.inc > /dev/null 2>&1" >> ~/.bashrc && \
    source ~/google-cloud-sdk/path.bash.inc >> {_GCLOUD_INSTALLATION_LOG} 2>&1; }}; }} && \
    popd &>/dev/null'

# TODO(zhwu): Move the default AMI size to the catalog instead.
DEFAULT_GCP_IMAGE_GB = 50

# Firewall rule name for user opened ports.
USER_PORTS_FIREWALL_RULE_NAME = 'sky-ports-{}'

# UX message when image not found in GCP.
# pylint: disable=line-too-long
_IMAGE_NOT_FOUND_UX_MESSAGE = (
    'Image {image_id!r} not found in GCP.\n'
    '\nTo find GCP images: https://cloud.google.com/compute/docs/images\n'
    f'Format: {colorama.Style.BRIGHT}projects/<project-id>/global/images/<image-name>{colorama.Style.RESET_ALL}\n'
    'Example: projects/deeplearning-platform-release/global/images/common-cpu-v20230615-debian-11-py310\n'
    '\nTo find machine images: https://cloud.google.com/compute/docs/machine-images\n'
    f'Format: {colorama.Style.BRIGHT}projects/<project-id>/global/machineImages/<machine-image-name>{colorama.Style.RESET_ALL}\n'
    f'\nYou can query image id using: {colorama.Style.BRIGHT}gcloud compute images list --project <project-id> --no-standard-images{colorama.Style.RESET_ALL}'
    f'\nTo query common AI images: {colorama.Style.BRIGHT}gcloud compute images list --project deeplearning-platform-release | less{colorama.Style.RESET_ALL}'
)

# Image ID tags
_DEFAULT_CPU_IMAGE_ID = 'skypilot:custom-cpu-ubuntu-2204'
# For GPU-related package version, see sky/catalog/images/provisioners/cuda.sh
# Default GPU image: NVIDIA 580 open kernel module + CUDA 13. Supports Turing
# and later only.
_DEFAULT_GPU_IMAGE_ID = 'skypilot:custom-gpu-ubuntu-2204-cuda13'
# Legacy GPU image: NVIDIA 535 proprietary driver + CUDA 12. Used for pre-Turing
# GPUs (V100, P100, P4, M60) that the open kernel module does not support.
_DEFAULT_GPU_CUDA12_IMAGE_ID = 'skypilot:custom-gpu-ubuntu-2204'
_DEFAULT_GPU_K80_IMAGE_ID = 'skypilot:k80-debian-10'
_DEFAULT_TPU_V6E_IMAGE_ID = (
    'projects/ubuntu-os-accelerator-images/global/images/family/'
    'ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e')
_DEFAULT_TPU7X_IMAGE_ID = (
    'projects/ubuntu-os-accelerator-images/global/images/family/'
    'ubuntu-accel-2404-amd64-tpu-tpu7x')
_COMPUTE_TPU_MACHINE_TYPE_PREFIXES = ('ct6e-standard-', 'tpu7x-standard-')
_CT6E_EIGHT_CHIP_MACHINE_TYPES = frozenset({
    'ct6e-standard-8t',
    'ct6e-standard-8t-tpu',
})
_COMPUTE_TPU_FLEX_START_PRICE_FACTOR = 0.5
_COMPUTE_TPU_FLEX_START_ZONES = {
    _COMPUTE_TPU_MACHINE_TYPE_PREFIXES[0]: {
        'asia-northeast1-b',
        'us-east5-a',
        'us-south1-ai1b',
    },
    _COMPUTE_TPU_MACHINE_TYPE_PREFIXES[1]: {'us-central1-c'},
}


def _is_compute_tpu_instance_type(instance_type: Optional[str]) -> bool:
    return (instance_type is not None and
            instance_type.startswith(_COMPUTE_TPU_MACHINE_TYPE_PREFIXES))


def _is_managed_instance_group_eligible(
        resources: 'resources.Resources') -> bool:
    return (_is_compute_tpu_instance_type(resources.instance_type) or
            (resources.accelerators is not None and
             not gcp_utils.is_tpu(resources)))


# Use COS image with GPU Direct support.
# Need to contact GCP support to build our own image for GPUDirect-TCPX support.
# Refer to https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/machine-learning/a3-highgpu-8g/README.md#before-starting
_DEFAULT_GPU_DIRECT_IMAGE_ID = 'skypilot:gpu-direct-cos'


def _run_output(cmd):
    proc = subprocess.run(cmd,
                          shell=True,
                          check=True,
                          stderr=subprocess.PIPE,
                          stdout=subprocess.PIPE)
    return proc.stdout.decode('ascii')


@annotations.ttl_cache(scope='request', timer=time.time, maxsize=1, ttl=5)
def _get_default():
    # pylint: disable=import-outside-toplevel
    import google.auth

    return google.auth.default()


@annotations.ttl_cache(scope='request', timer=time.time, maxsize=10, ttl=5)
def _list_enabled_services(project_id: str) -> Set[str]:
    # requires serviceusage.services.list
    proc = subprocess.run(
        f'gcloud services list --project {project_id} '
        '--format="value(config.name)"',
        check=True,
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE)
    return set(proc.stdout.decode().strip().splitlines())


def is_api_disabled(endpoint: str, project_id: str) -> bool:
    try:
        enabled = _list_enabled_services(project_id)
    except subprocess.CalledProcessError:
        return True
    return f'{endpoint}.googleapis.com' not in enabled


class GCPIdentityType(enum.Enum):
    """Type of the Application Default Credentials used by GCP APIs."""

    # Keep the legacy values as well as the legacy member alias: this enum is
    # internal, but plugins may still construct it by value.
    AUTHORIZED_USER = ''
    SHARED_CREDENTIALS_FILE = ''
    SERVICE_ACCOUNT = 'iam.gserviceaccount.com'
    METADATA_SERVICE_ACCOUNT = 'metadata_service_account'
    EXTERNAL_ACCOUNT = 'external_account'
    IMPERSONATED_SERVICE_ACCOUNT = 'impersonated_service_account'
    UNKNOWN = 'unknown'

    def can_credential_expire(self) -> bool:
        # Metadata and service-account key credentials can continuously mint
        # access tokens without an interactive user session.  For all other
        # credential chains, be conservative: authorized-user sessions can be
        # revoked/expire, external providers can stop issuing subject tokens,
        # and impersonated credentials can inherit either kind of source.
        return self not in {
            GCPIdentityType.SERVICE_ACCOUNT,
            GCPIdentityType.METADATA_SERVICE_ACCOUNT,
        }


@registry.CLOUD_REGISTRY.register
class GCP(clouds.Cloud):
    """Google Cloud Platform."""

    _REPR = 'GCP'

    # GCP has a 63 char limit; however, Ray autoscaler adds many
    # characters. Through testing, this is the maximum length for the Sky
    # cluster name on GCP.  Ref:
    # https://cloud.google.com/compute/docs/naming-resources#resource-name-format
    # NOTE: actually 37 is maximum for a single-node cluster which gets the
    # suffix '-head', but 35 for a multinode cluster because workers get the
    # suffix '-worker'. Here we do not distinguish these cases and take the
    # lower limit.
    _MAX_CLUSTER_NAME_LEN_LIMIT = 35

    _SUPPORTS_SERVICE_ACCOUNT_ON_REMOTE = True

    _INDENT_PREFIX = '    '
    _DEPENDENCY_HINT = (
        'GCP tools are not installed. Run the following commands:\n'
        # Install the Google Cloud SDK:
        f'{_INDENT_PREFIX}  $ pip install google-api-python-client\n'
        f'{_INDENT_PREFIX}  $ conda install -c conda-forge '
        'google-cloud-sdk -y\n'
        f'{_INDENT_PREFIX} If gcloud was recently installed with wget, API server'
        ' may need to be restarted with following commands:\n'
        f'{_INDENT_PREFIX}  $ sky api stop; sky api start')

    _CREDENTIAL_HINT = (
        'Run the following commands:\n'
        # This authenticates the CLI to make `gsutil` work:
        f'{_INDENT_PREFIX}  $ gcloud init\n'
        # This will generate
        # ~/.config/gcloud/application_default_credentials.json.
        f'{_INDENT_PREFIX}  $ gcloud auth application-default login\n'
        f'{_INDENT_PREFIX}For more info: '
        'https://docs.skypilot.co/en/latest/getting-started/installation.html#google-cloud-platform-gcp'  # pylint: disable=line-too-long
    )
    _APPLICATION_CREDENTIAL_HINT = (
        'Run the following commands:\n'
        f'{_INDENT_PREFIX}  $ gcloud auth application-default login\n'
        f'{_INDENT_PREFIX}Or set the environment variable GOOGLE_APPLICATION_CREDENTIALS '
        'to the path of your service account key file.\n'
        f'{_INDENT_PREFIX}For more info: '
        'https://docs.skypilot.co/en/latest/getting-started/installation.html#google-cloud-platform-gcp'  # pylint: disable=line-too-long
    )

    _SUPPORTED_DISK_TIERS = set(resources_utils.DiskTier)
    PROVISIONER_VERSION = clouds.ProvisionerVersion.SKYPILOT
    STATUS_VERSION = clouds.StatusVersion.SKYPILOT

    @classmethod
    def _unsupported_features_for_resources(
        cls,
        resources: 'resources.Resources',
        region: Optional[str] = None,
    ) -> Dict[clouds.CloudImplementationFeatures, str]:
        unsupported = {}
        is_compute_tpu = _is_compute_tpu_instance_type(resources.instance_type)
        if gcp_utils.is_tpu_vm_pod(resources) or is_compute_tpu:
            unsupported = {
                clouds.CloudImplementationFeatures.STOP:
                    ('TPU VMs cannot be stopped. Delete them instead. Please '
                     'refer to: https://cloud.google.com/tpu/docs/'
                     'managing-tpus-compute'),
                clouds.CloudImplementationFeatures.AUTOSTOP:
                    ('TPU VMs cannot be autostopped. Use autodown instead.'),
            }
        if gcp_utils.is_tpu(resources) and not gcp_utils.is_tpu_vm(resources):
            # TPU node does not support multi-node.
            unsupported[clouds.CloudImplementationFeatures.MULTI_NODE] = (
                'TPU node does not support multi-node. Please set '
                'num_nodes to 1.')
        # TODO(zhwu): We probably need to store the MIG requirement in resources
        # because `skypilot_config` may change for an existing cluster.
        # Clusters created with MIG cannot be stopped.
        if (skypilot_config.get_effective_region_config(
                cloud='gcp',
                region=resources.region,
                keys=('managed_instance_group',),
                override_configs=resources.cluster_config_overrides) is not None
                and _is_managed_instance_group_eligible(resources)):
            unsupported[clouds.CloudImplementationFeatures.STOP] = (
                'Managed Instance Group (MIG) does not support stopping yet.')
            unsupported[clouds.CloudImplementationFeatures.AUTOSTOP] = (
                'Managed Instance Group (MIG) does not support autostop. Use '
                'autodown instead.')
            unsupported[clouds.CloudImplementationFeatures.SPOT_INSTANCE] = (
                'Managed Instance Group with DWS does not support '
                'spot instances.')

        unsupported[
            clouds.CloudImplementationFeatures.
            HIGH_AVAILABILITY_CONTROLLERS] = (
                f'High availability controllers are not supported on {cls._REPR}.'
            )
        unsupported[clouds.CloudImplementationFeatures.
                    LOCAL_DISK] = f'Local disk is not supported on {cls._REPR}'

        return unsupported

    @classmethod
    def max_cluster_name_length(cls) -> Optional[int]:
        return cls._MAX_CLUSTER_NAME_LEN_LIMIT

    #### Regions/Zones ####
    @classmethod
    def regions_with_offering(
        cls,
        instance_type: str,
        accelerators: Optional[Dict[str, int]],
        use_spot: bool,
        region: Optional[str],
        zone: Optional[str],
        resources: Optional['resources.Resources'] = None,
    ) -> List[clouds.Region]:
        if accelerators is None:
            regions = catalog.get_region_zones_for_instance_type(instance_type,
                                                                 use_spot,
                                                                 clouds='gcp')
        else:
            assert len(accelerators) == 1, accelerators
            acc = list(accelerators.keys())[0]
            acc_count = list(accelerators.values())[0]
            acc_regions = catalog.get_region_zones_for_accelerators(
                acc, acc_count, use_spot, clouds='gcp')
            if instance_type is None:
                regions = acc_regions
            elif instance_type == 'TPU-VM':
                regions = acc_regions
            else:
                vm_regions = catalog.get_region_zones_for_instance_type(
                    instance_type, use_spot, clouds='gcp')
                # Find the intersection between `acc_regions` and `vm_regions`.
                regions = []
                for r1 in acc_regions:
                    for r2 in vm_regions:
                        if r1.name != r2.name:
                            continue
                        assert r1.zones is not None, r1
                        assert r2.zones is not None, r2
                        zones = []
                        for z1 in r1.zones:
                            for z2 in r2.zones:
                                if z1.name == z2.name:
                                    zones.append(z1)
                        if zones:
                            regions.append(r1.set_zones(zones))
                        break

        if (resources is not None and
                instance_type.startswith(_COMPUTE_TPU_MACHINE_TYPE_PREFIXES)):
            machine_type_prefix = next(
                prefix for prefix in _COMPUTE_TPU_MACHINE_TYPE_PREFIXES
                if instance_type.startswith(prefix))
            flex_start_zones = _COMPUTE_TPU_FLEX_START_ZONES[
                machine_type_prefix]
            filtered_regions = []
            for offered_region in regions:
                managed_instance_group_config = (
                    skypilot_config.get_effective_region_config(
                        cloud='gcp',
                        region=offered_region.name,
                        keys=('managed_instance_group',),
                        default_value=None,
                        override_configs=resources.cluster_config_overrides))
                if managed_instance_group_config is None:
                    filtered_regions.append(offered_region)
                    continue
                assert offered_region.zones is not None, offered_region
                zones = [
                    offered_zone for offered_zone in offered_region.zones
                    if offered_zone.name in flex_start_zones
                ]
                if zones:
                    filtered_regions.append(offered_region.set_zones(zones))
            regions = filtered_regions

        if region is not None:
            regions = [r for r in regions if r.name == region]
        if zone is not None:
            for r in regions:
                assert r.zones is not None, r
                r.set_zones([z for z in r.zones if z.name == zone])
            regions = [r for r in regions if r.zones]
        return regions

    @classmethod
    def optimize_by_zone(cls) -> bool:
        return True

    @classmethod
    def zones_provision_loop(
        cls,
        *,
        region: str,
        num_nodes: int,
        instance_type: str,
        accelerators: Optional[Dict[str, int]] = None,
        use_spot: bool = False,
    ) -> Iterator[List[clouds.Zone]]:
        del num_nodes  # Unused.
        regions = cls.regions_with_offering(instance_type,
                                            accelerators,
                                            use_spot,
                                            region=region,
                                            zone=None)
        # GCP provisioner currently takes 1 zone per request.
        for r in regions:
            assert r.zones is not None, r
            for zone in r.zones:
                yield [zone]

    @classmethod
    def get_zone_shell_cmd(cls) -> Optional[str]:
        # The command for getting the current zone is from:
        # https://cloud.google.com/compute/docs/metadata/querying-metadata
        command_str = (
            'curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone'  # pylint: disable=line-too-long
            ' -H "Metadata-Flavor: Google" | awk -F/ \'{print $4}\'')
        return command_str

    #### Normal methods ####

    def instance_type_to_hourly_cost(self,
                                     instance_type: str,
                                     use_spot: bool,
                                     region: Optional[str] = None,
                                     zone: Optional[str] = None) -> float:
        return catalog.get_hourly_cost(instance_type,
                                       use_spot=use_spot,
                                       region=region,
                                       zone=zone,
                                       clouds='gcp')

    def resources_to_hourly_cost(self, resources: 'resources.Resources',
                                 region: Optional[str],
                                 zone: Optional[str]) -> float:
        hourly_cost = super().resources_to_hourly_cost(resources, region, zone)
        if not _is_compute_tpu_instance_type(resources.instance_type):
            return hourly_cost
        managed_instance_group_config = (
            skypilot_config.get_effective_region_config(
                cloud='gcp',
                region=region,
                keys=('managed_instance_group',),
                default_value=None,
                override_configs=resources.cluster_config_overrides))
        if managed_instance_group_config is None:
            return hourly_cost
        # Google's published DWS Flex-start price is 50% of on-demand for the
        # Compute TPU generations currently exposed here.  Keep the base
        # catalog price on-demand because these machine types can also launch
        # without a managed instance group.
        return hourly_cost * _COMPUTE_TPU_FLEX_START_PRICE_FACTOR

    def accelerators_to_hourly_cost(self,
                                    accelerators: Dict[str, int],
                                    use_spot: bool,
                                    region: Optional[str] = None,
                                    zone: Optional[str] = None) -> float:
        assert len(accelerators) == 1, accelerators
        acc, acc_count = list(accelerators.items())[0]
        return catalog.get_accelerator_hourly_cost(acc,
                                                   acc_count,
                                                   use_spot=use_spot,
                                                   region=region,
                                                   zone=zone,
                                                   clouds='gcp')

    def get_egress_cost(self, num_gigabytes: float):
        # In general, query this from the cloud:
        #   https://cloud.google.com/storage/pricing#network-pricing
        # NOTE: egress to worldwide (excl. China, Australia).
        if num_gigabytes <= 1024:
            return 0.12 * num_gigabytes
        elif num_gigabytes <= 1024 * 10:
            return 0.11 * num_gigabytes
        else:
            return 0.08 * num_gigabytes

    @classmethod
    def _is_machine_image(cls, image_id: str) -> bool:
        find_machine = re.match(r'projects/.*/.*/machineImages/.*', image_id)
        return find_machine is not None

    @classmethod
    @annotations.lru_cache(scope='global', maxsize=1)
    def _get_image_size(cls, image_id: str) -> float:
        if image_id.startswith('skypilot:'):
            return DEFAULT_GCP_IMAGE_GB
        try:
            compute = gcp.build('compute',
                                'v1',
                                credentials=None,
                                cache_discovery=False)
        except gcp.credential_error_exception():
            return DEFAULT_GCP_IMAGE_GB
        try:
            image_attrs = image_id.split('/')
            if len(image_attrs) == 1:
                with ux_utils.print_exception_no_traceback():
                    raise ValueError(
                        _IMAGE_NOT_FOUND_UX_MESSAGE.format(image_id=image_id))
            project = image_attrs[1]
            image_name = image_attrs[-1]
            # We support both GCP's Machine Images and Custom Images, both
            # of which are specified with the image_id field. We will
            # distinguish them by checking if the image_id contains
            # 'machineImages'.
            if cls._is_machine_image(image_id):
                image_infos = compute.machineImages().get(
                    project=project, machineImage=image_name).execute()
                # The VM launching in a different region than the machine
                # image is supported by GCP, so we do not need to check the
                # storageLocations.
                return float(
                    image_infos['instanceProperties']['disks'][0]['diskSizeGb'])
            else:
                start = time.time()
                image_infos = compute.images().get(project=project,
                                                   image=image_name).execute()
                logger.debug(f'GCP image get took {time.time() - start:.2f}s')
                return float(image_infos['diskSizeGb'])
        except gcp.http_error_exception() as e:
            if e.resp.status == 403:
                with ux_utils.print_exception_no_traceback():
                    raise ValueError('Not able to access the image '
                                     f'{image_id!r}') from None
            if e.resp.status == 404:
                with ux_utils.print_exception_no_traceback():
                    raise ValueError(
                        _IMAGE_NOT_FOUND_UX_MESSAGE.format(
                            image_id=image_id)) from None
            raise

    @classmethod
    def get_image_size(cls, image_id: str, region: Optional[str]) -> float:
        del region  # Unused.
        return cls._get_image_size(image_id)

    @classmethod
    def get_default_instance_type(
        cls,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        disk_tier: Optional[resources_utils.DiskTier] = None,
        local_disk: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        use_spot: bool = False,
        max_hourly_cost: Optional[float] = None,
    ) -> Optional[str]:
        return catalog.get_default_instance_type(
            cpus=cpus,
            memory=memory,
            disk_tier=disk_tier,
            local_disk=local_disk,
            region=region,
            zone=zone,
            use_spot=use_spot,
            max_hourly_cost=max_hourly_cost,
            clouds='gcp')

    @classmethod
    def failover_disk_tier(
        cls, instance_type: Optional[str],
        disk_tier: Optional[resources_utils.DiskTier]
    ) -> Optional[resources_utils.DiskTier]:
        if (disk_tier is not None and
                disk_tier != resources_utils.DiskTier.BEST):
            return disk_tier
        # Failover disk tier from ultra to low.
        all_tiers = list(reversed(resources_utils.DiskTier))
        start_index = all_tiers.index(GCP._translate_disk_tier(disk_tier))
        while start_index < len(all_tiers):
            disk_tier = all_tiers[start_index]
            ok, _ = GCP.check_disk_tier(instance_type, disk_tier)
            if ok:
                return disk_tier
            start_index += 1
        assert False, 'Low disk tier should always be supported on GCP.'

    @staticmethod
    def _get_gpu_image_id(acc: str) -> str:
        """Returns the default image tag for a (non-GPU-direct) GPU."""
        if acc == 'K80':
            # Though the image is called cu113, it actually has later
            # versions of CUDA as noted below.
            # CUDA driver version 470.57.02, CUDA Library 11.4
            return _DEFAULT_GPU_K80_IMAGE_ID
        if gpu_utils.is_legacy_driver_gpu(acc):
            # Pre-Turing GPUs (V100, P100, P4, M60) are not supported by the
            # open kernel module in the default image.
            # CUDA driver version 535, CUDA Library 12.
            return _DEFAULT_GPU_CUDA12_IMAGE_ID
        # CUDA driver version 580 (open), CUDA Library 13.
        return _DEFAULT_GPU_IMAGE_ID

    def make_deploy_resources_variables(
        self,
        resources: 'resources.Resources',
        cluster_name: resources_utils.ClusterName,
        region: 'clouds.Region',
        zones: Optional[List['clouds.Zone']],
        num_nodes: int,
        dryrun: bool = False,
        volume_mounts: Optional[List['volume_lib.VolumeMount']] = None,
    ) -> Dict[str, Optional[str]]:
        assert zones is not None, (region, zones)

        region_name = region.name
        zone_name = zones[0].name

        # gcloud compute images list \
        # --project deeplearning-platform-release \
        # --no-standard-images
        # We use the debian image, as the ubuntu image has some connectivity
        # issue when first booted.
        image_id = _DEFAULT_CPU_IMAGE_ID

        r = resources
        if r.instance_type is not None:
            if r.instance_type.startswith(
                    _COMPUTE_TPU_MACHINE_TYPE_PREFIXES[0]):
                image_id = _DEFAULT_TPU_V6E_IMAGE_ID
            elif r.instance_type.startswith(
                    _COMPUTE_TPU_MACHINE_TYPE_PREFIXES[1]):
                image_id = _DEFAULT_TPU7X_IMAGE_ID
        # Find GPU spec, if any.
        is_compute_tpu = _is_compute_tpu_instance_type(r.instance_type)
        resources_vars = {
            'instance_type': r.instance_type,
            'region': region_name,
            'zones': zone_name,
            'gpu': None,
            'gpu_count': None,
            'tpu': None,
            'tpu_vm': False,
            'custom_resources': None,
            'use_spot': r.use_spot,
            'is_compute_tpu': is_compute_tpu,
            # CT6e 8-chip VMs require single-threaded cores.  Keep this in
            # node_config so both direct VM and regional MIG launches use the
            # same Compute Engine setting.
            'threads_per_core': 1 if r.instance_type
                                in _CT6E_EIGHT_CHIP_MACHINE_TYPES else None,
            'gcp_project_id': self.get_project_id(dryrun),
            **GCP._get_disk_specs(
                r.instance_type,
                GCP.failover_disk_tier(r.instance_type, r.disk_tier)),
        }
        docker_run_options = ['--privileged'] if is_compute_tpu else []
        enable_gpu_direct = skypilot_config.get_effective_region_config(
            cloud='gcp',
            region=region_name,
            keys=('enable_gpu_direct',),
            default_value=False,
            override_configs=resources.cluster_config_overrides)
        resources_vars['enable_gpu_direct'] = enable_gpu_direct
        network_tier = (r.network_tier if r.network_tier is not None else
                        resources_utils.NetworkTier.STANDARD)
        resources_vars['network_tier'] = network_tier.value
        accelerators = r.accelerators
        if accelerators is not None:
            assert len(accelerators) == 1, r
            acc, acc_count = list(accelerators.items())[0]
            resources_vars['custom_resources'] = json.dumps(accelerators,
                                                            separators=(',',
                                                                        ':'))
            if 'tpu' in acc:
                resources_vars['tpu_type'] = acc.replace('tpu-', '')
                assert r.accelerator_args is not None, r

                resources_vars['tpu_vm'] = r.accelerator_args.get(
                    'tpu_vm', True)
                resources_vars['runtime_version'] = r.accelerator_args[
                    'runtime_version']
                resources_vars['tpu_node_name'] = r.accelerator_args.get(
                    'tpu_name')
                resources_vars['gcp_queued_resource'] = r.accelerator_args.get(
                    'gcp_queued_resource')
                # TPU VMs require privileged mode for docker containers to
                # access TPU devices.
                docker_run_options.append('--privileged')
            else:
                # Convert to GCP names:
                # https://cloud.google.com/compute/docs/gpus
                if acc in ('A100-80GB', 'L4', 'B200'):
                    # A100-80GB, L4, and B200 use the `nvidia-<acc>` form
                    # rather than `nvidia-tesla-<acc>`.
                    resources_vars['gpu'] = f'nvidia-{acc.lower()}'
                elif acc == 'RTXPRO6000':
                    # GCP's accelerator type is `nvidia-rtx-pro-6000`; the
                    # `nvidia-{acc.lower()}` shortcut would produce the wrong
                    # `nvidia-rtxpro6000`.
                    resources_vars['gpu'] = 'nvidia-rtx-pro-6000'
                elif acc in ('H100', 'H100-MEGA'):
                    resources_vars['gpu'] = f'nvidia-{acc.lower()}-80gb'
                elif acc in ('H200',):
                    resources_vars['gpu'] = f'nvidia-{acc.lower()}-141gb'
                else:
                    resources_vars['gpu'] = 'nvidia-tesla-{}'.format(
                        acc.lower())
                resources_vars['gpu_count'] = acc_count
                if enable_gpu_direct or network_tier == resources_utils.NetworkTier.BEST:
                    # The actual image id is set in resources.py (see _try_validate_image_id)
                    # and reference GCP_GPU_DIRECT_IMAGE_ID
                    image_id = _DEFAULT_GPU_DIRECT_IMAGE_ID
                else:
                    image_id = GCP._get_gpu_image_id(acc)

        cloud_image_id = resources.get_cloud_image_id()
        if cloud_image_id is not None:
            if None in cloud_image_id:
                image_id = cloud_image_id[None]
            else:
                assert region_name in cloud_image_id, cloud_image_id
                image_id = cloud_image_id[region_name]
        if image_id.startswith('skypilot:'):
            image_id = catalog.get_image_id_from_tag(image_id, clouds='gcp')

        assert image_id is not None, (image_id, r)
        resources_vars['image_id'] = image_id
        resources_vars['machine_image'] = None

        if self._is_machine_image(image_id):
            resources_vars['machine_image'] = image_id
            resources_vars['image_id'] = None

        firewall_rule = None
        if resources.ports is not None:
            firewall_rule = (USER_PORTS_FIREWALL_RULE_NAME.format(
                cluster_name.name_on_cloud))
        resources_vars['firewall_rule'] = firewall_rule

        # For TPU nodes. TPU VMs do not need TPU_NAME.
        tpu_node_name = resources_vars.get('tpu_node_name')
        if gcp_utils.is_tpu(resources) and not gcp_utils.is_tpu_vm(resources):
            if tpu_node_name is None:
                tpu_node_name = cluster_name.name_on_cloud

        resources_vars['tpu_node_name'] = tpu_node_name

        managed_instance_group_config = skypilot_config.get_effective_region_config(
            cloud='gcp',
            region=region_name,
            keys=('managed_instance_group',),
            default_value=None,
            override_configs=resources.cluster_config_overrides)
        use_mig = (managed_instance_group_config is not None and
                   _is_managed_instance_group_eligible(r))
        resources_vars['gcp_use_managed_instance_group'] = use_mig
        resources_vars['gcp_is_tpu_mig'] = use_mig and is_compute_tpu
        # Convert boolean to 0 or 1 in string, as GCP does not support boolean
        # value in labels for TPU VM APIs.
        resources_vars['gcp_use_managed_instance_group_value'] = str(
            int(use_mig))
        if use_mig:
            resources_vars.update(managed_instance_group_config)
        resources_vars[
            'force_enable_external_ips'] = skypilot_config.get_effective_region_config(
                cloud='gcp',
                region=region_name,
                keys=('force_enable_external_ips',),
                default_value=False)

        volumes, device_mount_points = GCP._get_volumes_specs(
            region, zones, r.instance_type, r.volumes, use_mig,
            resources_vars['tpu_vm'])
        resources_vars['volumes'] = volumes

        resources_vars['user_data'] = None
        user_data = ''
        if device_mount_points:
            # Build the device_mounts array
            device_mounts_array = []
            for device_name, mount_point in device_mount_points.items():
                device_mounts_array.append(f'["{device_name}"]="{mount_point}"')
                docker_run_options.append(
                    f'--volume={mount_point}:{mount_point}')
            device_mounts_str = '\n        '.join(device_mounts_array)

            # Format the template with the device_mounts array
            user_data += constants.DISK_MOUNT_USER_DATA_TEMPLATE.format(
                device_mounts=device_mounts_str)

        # Add gVNIC from config
        resources_vars[
            'enable_gvnic'] = skypilot_config.get_effective_region_config(
                cloud='gcp',
                region=region_name,
                keys=('enable_gvnic',),
                default_value=False,
                override_configs=resources.cluster_config_overrides)
        placement_policy = skypilot_config.get_effective_region_config(
            cloud='gcp',
            region=region_name,
            keys=('placement_policy',),
            default_value=None,
            override_configs=resources.cluster_config_overrides)
        if enable_gpu_direct or network_tier == resources_utils.NetworkTier.BEST:
            user_data += constants.GPU_DIRECT_TCPX_USER_DATA
            docker_run_options += constants.GPU_DIRECT_TCPX_SPECIFIC_OPTIONS
            if placement_policy is None:
                placement_policy = constants.COMPACT_GROUP_PLACEMENT_POLICY
        if user_data:
            resources_vars[
                'user_data'] = constants.BASH_SCRIPT_START + user_data
        if docker_run_options:
            resources_vars['docker_run_options'] = docker_run_options
        resources_vars['placement_policy'] = placement_policy

        return resources_vars

    def _get_feasible_launchable_resources(
        self, resources: 'resources.Resources'
    ) -> 'resources_utils.FeasibleResources':
        if resources.instance_type is not None:
            assert resources.is_launchable(), resources
            ok, _ = GCP.check_disk_tier(resources.instance_type,
                                        resources.disk_tier)
            if not ok:
                return resources_utils.FeasibleResources([], [], None)
            return resources_utils.FeasibleResources([resources], [], None)

        if resources.accelerators is None:
            # Return a default instance type with the given number of vCPUs.
            host_vm_type = GCP.get_default_instance_type(
                cpus=resources.cpus,
                memory=resources.memory,
                disk_tier=resources.disk_tier,
                local_disk=resources.local_disk,
                region=resources.region,
                zone=resources.zone,
                use_spot=resources.use_spot,
                max_hourly_cost=resources.max_hourly_cost)
            if host_vm_type is None:
                # TODO: Add hints to all return values in this method to help
                #  users understand why the resources are not launchable.
                return resources_utils.FeasibleResources([], [], None)
            ok, _ = GCP.check_disk_tier(host_vm_type, resources.disk_tier)
            if not ok:
                return resources_utils.FeasibleResources([], [], None)
            r = resources.copy(
                cloud=GCP(),
                instance_type=host_vm_type,
                accelerators=None,
                cpus=None,
                memory=None,
            )
            return resources_utils.FeasibleResources([r], [], None)

        # Find instance candidates to meet user's requirements
        assert len(resources.accelerators.items()
                  ) == 1, 'cannot handle more than one accelerator candidates.'
        acc, acc_count = list(resources.accelerators.items())[0]
        use_tpu_vm = gcp_utils.is_tpu_vm(resources)

        # For TPU VMs, the instance type is fixed to 'TPU-VM'. However, we still
        # need to call the below function to get the fuzzy candidate list.
        (instance_list,
         fuzzy_candidate_list) = catalog.get_instance_type_for_accelerator(
             acc,
             acc_count,
             cpus=resources.cpus if not use_tpu_vm else None,
             memory=resources.memory if not use_tpu_vm else None,
             use_spot=resources.use_spot,
             local_disk=resources.local_disk,
             region=resources.region,
             zone=resources.zone,
             max_hourly_cost=resources.max_hourly_cost,
             clouds='gcp')

        if instance_list is None:
            return resources_utils.FeasibleResources([], fuzzy_candidate_list,
                                                     None)
        assert len(
            instance_list
        ) == 1, f'More than one instance type matched, {instance_list}'

        if use_tpu_vm:
            host_vm_type = 'TPU-VM'
            # FIXME(woosuk, wei-lin): This leverages the fact that TPU VMs
            # have 96 vCPUs, and 240 vCPUs for tpu-v4. We need to move
            # this to service catalog, instead.
            num_cpus_in_tpu_vm = 240 if 'v4' in acc else 96
            if resources.cpus is not None:
                if resources.cpus.endswith('+'):
                    cpus = float(resources.cpus[:-1])
                    if cpus > num_cpus_in_tpu_vm:
                        return resources_utils.FeasibleResources(
                            [], fuzzy_candidate_list, None)
                else:
                    cpus = float(resources.cpus)
                    if cpus != num_cpus_in_tpu_vm:
                        return resources_utils.FeasibleResources(
                            [], fuzzy_candidate_list, None)
            # FIXME(woosuk, wei-lin): This leverages the fact that TPU VMs
            # have 334 GB RAM, and 400 GB RAM for tpu-v4. We need to move
            # this to service catalog, instead.
            memory_in_tpu_vm = 400 if 'v4' in acc else 334
            if resources.memory is not None:
                if resources.memory.endswith('+'):
                    memory = float(resources.memory[:-1])
                    if memory > memory_in_tpu_vm:
                        return resources_utils.FeasibleResources(
                            [], fuzzy_candidate_list, None)
                else:
                    memory = float(resources.memory)
                    if memory != memory_in_tpu_vm:
                        return resources_utils.FeasibleResources(
                            [], fuzzy_candidate_list, None)
        else:
            host_vm_type = instance_list[0]

        ok, _ = GCP.check_disk_tier(host_vm_type, resources.disk_tier)
        if not ok:
            return resources_utils.FeasibleResources([], fuzzy_candidate_list,
                                                     None)
        acc_dict = {acc: acc_count}
        r = resources.copy(
            cloud=GCP(),
            instance_type=host_vm_type,
            accelerators=acc_dict,
            cpus=None,
            memory=None,
        )
        return resources_utils.FeasibleResources([r], fuzzy_candidate_list,
                                                 None)

    @classmethod
    def get_accelerators_from_instance_type(
        cls,
        instance_type: str,
    ) -> Optional[Dict[str, Union[int, float]]]:
        # GCP handles accelerators separately from regular instance types.
        # This method supports automatically inferring the GPU type for
        # the instance type that come with GPUs pre-attached.
        return catalog.get_accelerators_from_instance_type(instance_type,
                                                           clouds='gcp')

    @classmethod
    def get_vcpus_mem_from_instance_type(
        cls,
        instance_type: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        return catalog.get_vcpus_mem_from_instance_type(instance_type,
                                                        clouds='gcp')

    @classmethod
    def _find_application_key_path(cls) -> str:
        # Check the application default credentials in the environment variable.
        # If the file does not exist, fallback to the default path.
        application_key_path = os.environ.get(_GCP_APPLICATION_CREDENTIAL_ENV,
                                              None)
        if application_key_path is not None:
            if not os.path.isfile(os.path.expanduser(application_key_path)):
                raise FileNotFoundError(
                    f'{_GCP_APPLICATION_CREDENTIAL_ENV}={application_key_path},'
                    ' but the file does not exist.')
            return application_key_path
        application_key_path = os.path.join(
            cls._get_local_gcloud_config_dir(),
            os.path.basename(DEFAULT_GCP_APPLICATION_CREDENTIAL_PATH))
        if not os.path.isfile(application_key_path):
            # Fallback to the default application credential path.
            raise FileNotFoundError(application_key_path)
        return application_key_path

    @staticmethod
    def _get_local_gcloud_config_dir(source_root: Optional[str] = None) -> str:
        """Resolve and validate the local gcloud configuration directory."""
        if source_root is None:
            source_root = os.environ.get('CLOUDSDK_CONFIG')
            if source_root is None:
                source_root = _GCLOUD_CONFIG_DIR
        if not source_root.strip():
            raise exceptions.CloudUserIdentityError(
                'The local gcloud configuration directory is empty. Check '
                '`CLOUDSDK_CONFIG` and retry.')
        source_root = os.path.abspath(
            os.path.expanduser(os.path.expandvars(source_root)))
        if not os.path.isdir(source_root):
            raise exceptions.CloudUserIdentityError(
                'The local gcloud configuration directory does not exist or '
                f'is not a directory: {source_root!r}. Check '
                '`CLOUDSDK_CONFIG` and retry.')
        return source_root

    @classmethod
    def _check_compute_credentials(
            cls) -> Tuple[bool, Optional[Union[str, Dict[str, str]]]]:
        """Checks if the user has access credentials to this cloud's compute service."""
        return cls._check_credentials(
            [
                ('compute', 'Compute Engine'),
                ('cloudresourcemanager', 'Cloud Resource Manager'),
                ('iam', 'Identity and Access Management (IAM)'),
                ('tpu', 'Cloud TPU'),  # Keep as final element.
            ],
            gcp_utils.get_minimal_compute_permissions())

    @classmethod
    def _check_storage_credentials(
            cls) -> Tuple[bool, Optional[Union[str, Dict[str, str]]]]:
        """Checks if the user has access credentials to this cloud's storage service."""
        return cls._check_credentials(
            [('storage', 'Cloud Storage')],
            gcp_utils.get_minimal_storage_permissions())

    @classmethod
    def _check_credentials(
            cls, apis: List[Tuple[str, str]],
            gcp_minimal_permissions: List[str]) -> Tuple[bool, Optional[str]]:
        """Checks if the user has access credentials to this cloud."""
        try:
            # pylint: disable=import-outside-toplevel,unused-import
            # Check google-api-python-client installation.
            from google import auth  # type: ignore
            import googleapiclient

            if shutil.which('gcloud') is None:
                raise RuntimeError('Missing `gcloud` cli dependency.')
        except (ImportError, RuntimeError) as e:
            return False, (
                f'{cls._DEPENDENCY_HINT}\n'
                f'{cls._INDENT_PREFIX}Credentials may also need to be set. '
                f'{cls._CREDENTIAL_HINT}\n'
                f'{cls._INDENT_PREFIX}Details: '
                f'{common_utils.format_exception(e, use_bracket=True)}')

        identity_type = cls._get_identity_type()
        if identity_type == GCPIdentityType.AUTHORIZED_USER:
            # These files are only required when using the shared credentials
            # to access GCP. They are not required when using service account.
            try:
                # These files are required because they will be synced to remote
                # VMs for `gsutil` to access private storage buckets.
                # `auth.default()` does not guarantee these files exist.
                gcloud_config_dir = cls._get_local_gcloud_config_dir()
                for filename in ('access_tokens.db', 'credentials.db'):
                    credential_path = os.path.join(gcloud_config_dir, filename)
                    if not os.path.isfile(credential_path):
                        raise FileNotFoundError(credential_path)
            except (FileNotFoundError, exceptions.CloudUserIdentityError) as e:
                return False, (
                    f'Credentials are not set. '
                    f'{cls._CREDENTIAL_HINT}\n'
                    f'{cls._INDENT_PREFIX}Details: '
                    f'{common_utils.format_exception(e, use_bracket=True)}')

            try:
                cls._find_application_key_path()
            except FileNotFoundError as e:
                return False, (
                    f'Application credentials are not set. '
                    f'{cls._APPLICATION_CREDENTIAL_HINT}\n'
                    f'{cls._INDENT_PREFIX}Details: '
                    f'{common_utils.format_exception(e, use_bracket=True)}')

        try:
            # Check if application default credentials are set.
            project_id = cls.get_project_id()

            # This identity is derived from the same ADC used for API calls.
            identity = cls.get_active_user_identity()
        except (auth.exceptions.DefaultCredentialsError,
                exceptions.CloudUserIdentityError) as e:
            # See also: https://stackoverflow.com/a/53307505/1165051
            return False, (
                'Getting project ID or user identity failed. You can debug '
                'with `gcloud auth list`. To fix this, '
                f'{cls._CREDENTIAL_HINT[0].lower()}'
                f'{cls._CREDENTIAL_HINT[1:]}\n'
                f'{cls._INDENT_PREFIX}Details: '
                f'{common_utils.format_exception(e, use_bracket=True)}')

        try:
            # Some GCP paths still invoke gcloud (for example Service Usage
            # and OS Login).  Do not allow those commands to silently run as a
            # different principal from the ADC-backed API clients.
            gcloud_account = cls._get_active_gcloud_account()
        except exceptions.CloudUserIdentityError as e:
            return False, (
                'Getting the effective gcloud account failed. '
                f'{cls._CREDENTIAL_HINT}\n'
                f'{cls._INDENT_PREFIX}Details: '
                f'{common_utils.format_exception(e, use_bracket=True)}')

        if gcloud_account is not None:
            assert identity is not None
            project_suffix = f' [project_id={project_id}]'
            adc_account = identity[0]
            if adc_account.endswith(project_suffix):
                adc_account = adc_account[:-len(project_suffix)]
            # Workforce owner identities include a pool/provider-bound subject
            # digest to avoid collisions. gcloud exposes only the verified
            # email for that credential, so compare that email while retaining
            # the full identity for SkyPilot ownership checks.
            workforce_suffix = ' [workforce_subject='
            gcloud_comparable_adc_account = adc_account
            if workforce_suffix in gcloud_comparable_adc_account:
                gcloud_comparable_adc_account = (
                    gcloud_comparable_adc_account.split(workforce_suffix,
                                                        maxsplit=1)[0])
            if (gcloud_account.casefold() !=
                    gcloud_comparable_adc_account.casefold()):
                return False, (
                    'The active gcloud account and Application Default '
                    'Credentials (ADC) authorize as different users:\n'
                    f'    gcloud: {gcloud_account}\n'
                    f'    ADC:    {adc_account}\n'
                    'SkyPilot uses ADC for Google Cloud API calls and the '
                    'gcloud credential store for CLI operations. Sign in both '
                    'as the same principal. For user credentials, run `gcloud '
                    'auth login` and `gcloud auth application-default login`. '
                    'For service-account credentials, activate the same '
                    'service account in gcloud.')

        # This takes user's credential info from "~/.config/gcloud/application_default_credentials.json".  # pylint: disable=line-too-long
        credentials, project = _get_default()
        crm = gcp.build('cloudresourcemanager',
                        'v1',
                        credentials=credentials,
                        cache_discovery=False)
        permissions = {'permissions': gcp_minimal_permissions}
        request = crm.projects().testIamPermissions(resource=project,
                                                    body=permissions)
        try:
            ret_permissions = request.execute().get('permissions', [])
        except gcp.gcp_auth_refresh_error_exception() as e:
            return False, common_utils.format_exception(e, use_bracket=True)

        diffs = set(gcp_minimal_permissions).difference(set(ret_permissions))
        if diffs:
            identity_str = identity[0] if identity else None
            return False, (
                'The following permissions are not enabled for the current '
                f'GCP identity ({identity_str}):\n    '
                f'{diffs}\n    '
                'For more details, visit: https://docs.skypilot.co/en/latest/cloud-setup/cloud-permissions/gcp.html')  # pylint: disable=line-too-long

        # This code must be executed after the iam check above,
        # as the check below for api enablement itself needs:
        # - serviceusage.services.enable
        # - serviceusage.services.list
        # iam permissions.
        enabled_api = False
        for endpoint, display_name in apis:
            if is_api_disabled(endpoint, project_id):
                # For 'compute': ~55-60 seconds for the first run. If already
                # enabled, ~1s. Other API endpoints take ~1-5s to enable.
                if endpoint == 'compute':
                    suffix = ' (free of charge; this may take a minute)'
                else:
                    suffix = ' (free of charge)'
                print(f'\nEnabling {display_name} API{suffix}...')
                t1 = time.time()
                # requires serviceusage.services.enable
                proc = subprocess.run(
                    f'gcloud services enable {endpoint}.googleapis.com '
                    f'--project {project_id}',
                    check=False,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
                if proc.returncode == 0:
                    enabled_api = True
                    print(f'Done. Took {time.time() - t1:.1f} secs.')
                elif endpoint != 'tpu':
                    print('Failed. Detailed output:')
                    print(proc.stdout.decode())
                    return False, (
                        f'{display_name} API is disabled. Please retry '
                        '`sky check` in a few minutes, or manually enable it.')
                else:
                    # TPU API failed. Should still enable GCP.
                    print('Failed to enable Cloud TPU API. '
                          'This can be ignored if you do not use TPUs. '
                          'Otherwise, please enable it manually.\n'
                          'Detailed output:')
                    print(proc.stdout.decode())

        if enabled_api:
            print('\nHint: Enabled GCP API(s) may take a few minutes to take '
                  'effect. If any SkyPilot commands/calls failed, retry after '
                  'some time.')

        return True, None

    @staticmethod
    def _stage_gcloud_database(source_path: str, destination_path: str,
                               table: str, columns: Tuple[str, ...],
                               accounts: Tuple[str, ...]) -> int:
        """Copy only selected account rows into a fresh SQLite database.

        Copying a gcloud database and deleting rows is insufficient because
        SQLite freelist pages can retain deleted refresh tokens. Build a new
        database so credentials for inactive local accounts never enter the
        staged file.
        """
        placeholders = ','.join('?' for _ in accounts)
        column_list = ', '.join(columns)
        source_uri = f'file:{source_path}?mode=ro'
        with sqlite3.connect(source_uri, uri=True) as source:
            rows = source.execute(
                f'SELECT {column_list} FROM {table} '
                f'WHERE account_id IN ({placeholders})', accounts).fetchall()

        destination_dir = os.path.dirname(destination_path)
        os.makedirs(destination_dir, mode=0o700, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f'.{os.path.basename(destination_path)}.',
            dir=destination_dir)
        os.close(file_descriptor)
        try:
            with sqlite3.connect(temporary_path) as destination:
                if table == 'credentials':
                    destination.execute(
                        'CREATE TABLE credentials '
                        '(account_id TEXT PRIMARY KEY, value BLOB)')
                elif table == 'access_tokens':
                    destination.execute(
                        'CREATE TABLE access_tokens '
                        '(account_id TEXT PRIMARY KEY, access_token TEXT, '
                        'token_expiry TIMESTAMP, rapt_token TEXT, '
                        'id_token TEXT)')
                else:
                    raise ValueError(f'Unsupported gcloud table: {table!r}')
                if rows:
                    value_placeholders = ','.join('?' for _ in columns)
                    destination.executemany(
                        f'INSERT INTO {table} ({column_list}) '
                        f'VALUES ({value_placeholders})', rows)
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, destination_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        return len(rows)

    @classmethod
    def _stage_gcloud_cli_credentials(
            cls,
            base_account: str,
            effective_account: str,
            source_root: Optional[str] = None,
            staging_root: Optional[str] = None) -> Dict[str, str]:
        """Stage a single-principal gcloud configuration for remote upload."""
        source_root = cls._get_local_gcloud_config_dir(source_root)
        active_config = os.environ.get('CLOUDSDK_ACTIVE_CONFIG_NAME')
        if not active_config:
            active_config_path = os.path.join(source_root, 'active_config')
            active_config = 'default'
            if os.path.isfile(active_config_path):
                with open(active_config_path, encoding='utf-8') as file:
                    active_config = file.read().strip()
        if re.fullmatch(r'[A-Za-z0-9_-]+', active_config) is None:
            raise exceptions.CloudUserIdentityError(
                f'Invalid active gcloud configuration: {active_config!r}.')
        source_config = os.path.join(source_root, 'configurations',
                                     f'config_{active_config}')
        if not os.path.isfile(source_config):
            raise exceptions.CloudUserIdentityError(
                'The selected gcloud configuration file does not exist: '
                f'{source_config!r}. Check `CLOUDSDK_ACTIVE_CONFIG_NAME` and '
                'the active gcloud configuration, then retry.')

        staging_key = hashlib.sha256(
            (f'{os.path.realpath(source_root)}\0{active_config}\0'
             f'{base_account}\0{effective_account}').encode()).hexdigest()[:16]
        if staging_root is None:
            staging_root = os.path.expanduser(_STAGED_GCLOUD_CONFIG_DIR)
        staged_root = os.path.join(staging_root, staging_key)
        staged_configurations = os.path.join(staged_root, 'configurations')
        os.makedirs(staged_configurations, mode=0o700, exist_ok=True)

        parser = configparser.RawConfigParser()
        with open(source_config, encoding='utf-8') as file:
            parser.read_file(file)
        if not parser.has_section('core'):
            parser.add_section('core')
        parser.set('core', 'account', base_account)
        if not parser.has_section('auth'):
            parser.add_section('auth')
        for option in ('access_token_file', 'credential_file_override',
                       'login_config_file'):
            parser.remove_option('auth', option)
        impersonated_account = cls._get_configured_gcloud_impersonation()
        if impersonated_account is None:
            parser.remove_option('auth', 'impersonate_service_account')
        else:
            if (impersonated_account.casefold() !=
                    effective_account.casefold()):
                raise exceptions.CloudUserIdentityError(
                    'The configured gcloud impersonation target changed while '
                    'staging credentials. Re-run the command.')
            parser.set('auth', 'impersonate_service_account',
                       impersonated_account)

        staged_config = os.path.join(staged_configurations,
                                     f'config_{active_config}')
        file_descriptor, temporary_config = tempfile.mkstemp(
            prefix='.config.', dir=staged_configurations, text=True)
        try:
            with os.fdopen(file_descriptor, 'w', encoding='utf-8') as file:
                parser.write(file)
            os.chmod(temporary_config, 0o600)
            os.replace(temporary_config, staged_config)
        finally:
            if os.path.exists(temporary_config):
                os.remove(temporary_config)

        staged_active_config = os.path.join(staged_root, 'active_config')
        file_descriptor, temporary_active_config = tempfile.mkstemp(
            prefix='.active_config.', dir=staged_root, text=True)
        try:
            with os.fdopen(file_descriptor, 'w', encoding='utf-8') as file:
                # gcloud treats every byte in this file as part of the
                # configuration name; a trailing newline makes it look for
                # `config_default\n`.
                file.write(active_config)
            os.chmod(temporary_active_config, 0o600)
            os.replace(temporary_active_config, staged_active_config)
        finally:
            if os.path.exists(temporary_active_config):
                os.remove(temporary_active_config)

        mounts = {
            f'{_GCLOUD_CONFIG_DIR}/configurations': staged_configurations,
            f'{_GCLOUD_CONFIG_DIR}/active_config': staged_active_config,
        }
        database_specs = {
            'credentials.db': (
                'credentials',
                ('account_id', 'value'),
                (base_account,),
            ),
            'access_tokens.db': (
                'access_tokens',
                ('account_id', 'access_token', 'token_expiry', 'rapt_token',
                 'id_token'),
                tuple(dict.fromkeys((base_account, effective_account))),
            ),
        }
        for filename, (table, columns, accounts) in database_specs.items():
            source_path = os.path.join(source_root, filename)
            if not os.path.isfile(source_path):
                continue
            staged_path = os.path.join(staged_root, filename)
            row_count = cls._stage_gcloud_database(source_path, staged_path,
                                                   table, columns, accounts)
            if filename == 'credentials.db' and row_count == 0:
                raise exceptions.CloudUserIdentityError(
                    'The active gcloud account has no stored credential row. '
                    'Run `gcloud auth login` again before uploading local '
                    'credentials.')
            mounts[f'{_GCLOUD_CONFIG_DIR}/{filename}'] = staged_path
        if f'{_GCLOUD_CONFIG_DIR}/credentials.db' not in mounts:
            raise exceptions.CloudUserIdentityError(
                'The active gcloud account does not have a portable credential '
                'database. SkyPilot cannot upload ADC as a substitute because '
                'gcloud does not use ADC. Activate the same account with '
                '`gcloud auth login` (or `gcloud auth activate-service-account` '
                'for a service-account key), then retry.')
        return mounts

    def get_credential_file_mounts(self) -> Dict[str, str]:
        # Credential mount collection is called while inspecting all clouds.
        # Invalid or non-portable ADC therefore fails closed with no mounts;
        # `sky check gcp` provides the actionable authentication error.
        try:
            adc_credentials, _ = _get_default()
            identity_type = self._get_identity_type_from_credentials(
                adc_credentials)
        except Exception:  # pylint: disable=broad-except
            return {}

        portable_identity_types = {
            GCPIdentityType.AUTHORIZED_USER,
            GCPIdentityType.SERVICE_ACCOUNT,
            GCPIdentityType.IMPERSONATED_SERVICE_ACCOUNT,
        }
        if identity_type not in portable_identity_types:
            return {}
        if identity_type == GCPIdentityType.IMPERSONATED_SERVICE_ACCOUNT:
            source = getattr(adc_credentials, '_source_credentials', None)
            if (source is None or
                    self._get_identity_type_from_credentials(source)
                    == GCPIdentityType.EXTERNAL_ACCOUNT):
                # External-account source files and executable/URL providers
                # are not safely portable to an arbitrary remote VM.
                return {}

        adc_account = self._get_adc_principal(adc_credentials, identity_type)
        gcloud_account = self._get_active_gcloud_account()
        if gcloud_account.casefold() != adc_account.casefold():
            with ux_utils.print_exception_no_traceback():
                raise exceptions.CloudUserIdentityError(
                    'Refusing to upload GCP credentials because the effective '
                    f'gcloud account ({gcloud_account}) differs from ADC '
                    f'({adc_account}). Re-run `sky check gcp` after signing '
                    'both in as the same principal.')

        base_account = self._get_active_gcloud_base_account()
        credentials = self._stage_gcloud_cli_credentials(
            base_account, gcloud_account)
        try:
            application_key_path = self._find_application_key_path()
            # Upload ADC for SDK calls. The isolated gcloud store above makes
            # CLI and gsutil calls use the same effective principal without
            # exposing credentials for other locally cached accounts.
            credentials[DEFAULT_GCP_APPLICATION_CREDENTIAL_PATH] = (
                application_key_path)
        except FileNotFoundError:
            pass
        return credentials

    @annotations.lru_cache(scope='request', maxsize=1)
    def can_credential_expire(self) -> bool:
        try:
            credentials, _ = _get_default()
        except Exception:  # pylint: disable=broad-except
            return True
        return self._credentials_can_expire(credentials)

    @classmethod
    def _credentials_can_expire(cls,
                                credentials: Any,
                                _seen: Optional[Set[int]] = None) -> bool:
        """Whether a credential chain depends on a revocable user session."""
        if _seen is None:
            _seen = set()
        credential_id = id(credentials)
        if credential_id in _seen:
            return True
        _seen.add(credential_id)

        identity_type = cls._get_identity_type_from_credentials(credentials)
        if identity_type == GCPIdentityType.IMPERSONATED_SERVICE_ACCOUNT:
            source = getattr(credentials, '_source_credentials', None)
            if source is None:
                return True
            return cls._credentials_can_expire(source, _seen)
        # An unrecognized credential chain must not suppress warnings about
        # long-lived controllers.
        return identity_type.can_credential_expire()

    @classmethod
    def _get_identity_type(cls) -> Optional[GCPIdentityType]:
        try:
            credentials, _ = _get_default()
        except Exception:  # pylint: disable=broad-except
            return None
        return cls._get_identity_type_from_credentials(credentials)

    @staticmethod
    def _get_identity_type_from_credentials(
            credentials: Any) -> GCPIdentityType:
        """Classifies the credential object returned by google.auth.default().

        Class names are used instead of importing every google-auth credential
        module.  This keeps GCP an optional dependency and works for external
        account subclasses (AWS, identity-pool, and pluggable credentials).
        """
        credential_classes = {
            f'{base.__module__}.{base.__name__}'
            for base in type(credentials).__mro__
        }
        if ('google.oauth2.credentials.Credentials' in credential_classes):
            return GCPIdentityType.AUTHORIZED_USER
        if ('google.auth.external_account_authorized_user.Credentials'
                in credential_classes):
            return GCPIdentityType.EXTERNAL_ACCOUNT
        if ('google.auth.impersonated_credentials.Credentials'
                in credential_classes):
            return GCPIdentityType.IMPERSONATED_SERVICE_ACCOUNT
        if ('google.auth.external_account.Credentials' in credential_classes):
            return GCPIdentityType.EXTERNAL_ACCOUNT
        if (credential_classes.intersection({
                'google.auth.compute_engine.credentials.Credentials',
                'google.auth.app_engine.Credentials',
        })):
            return GCPIdentityType.METADATA_SERVICE_ACCOUNT
        if (credential_classes.intersection({
                'google.oauth2.service_account.Credentials',
                'google.oauth2.gdch_credentials.ServiceAccountCredentials',
        })):
            return GCPIdentityType.SERVICE_ACCOUNT
        return GCPIdentityType.UNKNOWN

    @classmethod
    def _get_adc_principal(cls, credentials: Any,
                           identity_type: GCPIdentityType) -> str:
        """Returns the effective principal represented by an ADC object."""

        def _validate_principal(principal: Any) -> str:
            if not isinstance(principal, str) or not principal:
                raise exceptions.CloudUserIdentityError(
                    'GCP returned an empty credential principal.')
            if (principal != principal.strip() or len(principal) > 2048 or
                    any(ord(char) < 32 for char in principal)):
                raise exceptions.CloudUserIdentityError(
                    'GCP returned an invalid credential principal.')
            return principal

        is_external_user = False
        if identity_type == GCPIdentityType.EXTERNAL_ACCOUNT:
            try:
                is_external_user = bool(getattr(credentials, 'is_user', False))
            except Exception:  # pylint: disable=broad-except
                is_external_user = False

        if (identity_type == GCPIdentityType.AUTHORIZED_USER or
                is_external_user):
            # The account field in an ADC JSON file is user-editable metadata;
            # verify the subject with an access token minted by the credential.
            try:
                oauth2 = gcp.build('oauth2',
                                   'v2',
                                   credentials=credentials,
                                   cache_discovery=False)
                user_info = oauth2.userinfo().get().execute(num_retries=3)
                verified_email = _validate_principal(user_info.get('email'))
                account = verified_email
                if is_external_user:
                    # Email alone is not globally unique across workforce
                    # pools. Bind the owner identity to the token subject and
                    # audience returned/used by this credential chain.
                    subject = _validate_principal(
                        user_info.get('id') or user_info.get('sub'))
                    audience = _validate_principal(
                        getattr(credentials, 'audience', None) or
                        getattr(credentials, '_audience', None))
                    subject_digest = hashlib.sha256(
                        f'{audience}\0{subject}'.encode()).hexdigest()
                    account = (f'{verified_email} '
                               f'[workforce_subject={subject_digest}]')
            except Exception as e:  # pylint: disable=broad-except
                with ux_utils.print_exception_no_traceback():
                    raise exceptions.CloudUserIdentityError(
                        'Application Default Credentials could not verify '
                        'their user with the Google OAuth userinfo endpoint. '
                        'Re-authenticate ADC with an email/userinfo scope or '
                        'use service-account impersonation.\n'
                        '  Reason: '
                        f'{common_utils.format_exception(e, use_bracket=True)}'
                    ) from e

            account_hint = getattr(credentials, 'account', None)
            if (isinstance(account_hint, str) and account_hint and
                    account_hint.casefold() != verified_email.casefold()):
                with ux_utils.print_exception_no_traceback():
                    raise exceptions.CloudUserIdentityError(
                        'The account recorded in the ADC file does not match '
                        'the OAuth subject. Recreate ADC with `gcloud auth '
                        'application-default login`.')
            return account

        service_account_email = getattr(credentials, 'service_account_email',
                                        None)
        if (isinstance(service_account_email, str) and
                service_account_email not in ('', 'default')):
            return _validate_principal(service_account_email)

        if identity_type == GCPIdentityType.METADATA_SERVICE_ACCOUNT:
            try:
                # Compute credentials initially expose the placeholder
                # "default"; refresh resolves the actual metadata identity.
                # google-auth is an optional GCP dependency.
                # pylint: disable=import-outside-toplevel
                from google.auth.transport import requests as auth_requests
                credentials.refresh(auth_requests.Request())
                service_account_email = credentials.service_account_email
            except Exception as e:  # pylint: disable=broad-except
                with ux_utils.print_exception_no_traceback():
                    raise exceptions.CloudUserIdentityError(
                        'Failed to resolve the GCP metadata service account.\n'
                        '  Reason: '
                        f'{common_utils.format_exception(e, use_bracket=True)}'
                    ) from e
            if service_account_email not in ('', 'default', None):
                return _validate_principal(service_account_email)

        if identity_type == GCPIdentityType.EXTERNAL_ACCOUNT:
            # A pool/provider audience is shared by multiple subjects and must
            # never be used as an owner identity.  Direct workload federation
            # does not expose a verified mapped subject through google-auth.
            with ux_utils.print_exception_no_traceback():
                raise exceptions.CloudUserIdentityError(
                    'SkyPilot cannot safely determine a unique principal for '
                    'direct external-account ADC. Configure service-account '
                    'impersonation (or a workforce user credential whose OAuth '
                    'userinfo includes both a stable subject and an email).')

        with ux_utils.print_exception_no_traceback():
            raise exceptions.CloudUserIdentityError(
                'SkyPilot could not determine the principal represented by '
                'Application Default Credentials '
                f'({identity_type.name.lower()}).')

    @classmethod
    def _get_configured_gcloud_impersonation(cls) -> Optional[str]:
        try:
            impersonated_account = _run_output(
                'gcloud config get-value auth/impersonate_service_account '
                '--quiet').strip()
        except subprocess.CalledProcessError:
            return None
        if impersonated_account and impersonated_account != '(unset)':
            return impersonated_account
        return None

    @classmethod
    def _get_active_gcloud_base_account(cls) -> str:
        """Returns the credentialed gcloud account before impersonation."""
        try:
            account = _run_output('gcloud auth list --filter=status:ACTIVE '
                                  '--format="value(account)"').strip()
        except subprocess.CalledProcessError as e:
            with ux_utils.print_exception_no_traceback():
                raise exceptions.CloudUserIdentityError(
                    'Failed to get the active gcloud account.\n'
                    '  Reason: '
                    f'{common_utils.format_exception(e, use_bracket=True)}'
                ) from e
        if not account:
            with ux_utils.print_exception_no_traceback():
                raise exceptions.CloudUserIdentityError(
                    'No GCP account is activated in gcloud. Run `gcloud auth '
                    'login` and verify `gcloud auth list '
                    '--filter=status:ACTIVE --format="value(account)"`.')
        return account

    @classmethod
    def _get_active_gcloud_account(cls) -> str:
        """Returns gcloud's effective account, including impersonation."""
        impersonated_account = cls._get_configured_gcloud_impersonation()
        if impersonated_account is not None:
            return impersonated_account
        return cls._get_active_gcloud_base_account()

    @classmethod
    def _get_legacy_impersonation_source_principal(
            cls, credentials: Any) -> Optional[str]:
        """Returns a verified pre-ADC owner identity for compatibility.

        Older SkyPilot versions stored the active source gcloud account even
        when API calls used an impersonated service account. Expose that source
        as a secondary identity only when google-auth provides and verifies the
        exact source credential object.
        """
        if (cls._get_identity_type_from_credentials(credentials) !=
                GCPIdentityType.IMPERSONATED_SERVICE_ACCOUNT):
            return None
        source = getattr(credentials, '_source_credentials', None)
        if source is None:
            return None
        source_type = cls._get_identity_type_from_credentials(source)
        if source_type not in {
                GCPIdentityType.AUTHORIZED_USER,
                GCPIdentityType.SERVICE_ACCOUNT,
                GCPIdentityType.METADATA_SERVICE_ACCOUNT,
        }:
            return None
        try:
            return cls._get_adc_principal(source, source_type)
        except exceptions.CloudUserIdentityError:
            return None

    @classmethod
    def get_user_identities(cls) -> List[List[str]]:
        """Returns the email address + project id of the active user."""
        gcp_workspace_config = json.dumps(
            skypilot_config.get_workspace_cloud('gcp'), sort_keys=True)
        return cls._get_user_identities(gcp_workspace_config)

    @classmethod
    @annotations.lru_cache(scope='request', maxsize=5)
    def _get_user_identities(
            cls, workspace_config: Optional[str]) -> List[List[str]]:
        # We add workspace_config in args to avoid caching the GCP identity
        # for when different workspace configs are used. Use json.dumps to
        # ensure the config is hashable.
        del workspace_config  # Unused

        try:
            credentials, _ = _get_default()
        except ImportError:
            # Preserve identity inspection in minimal installations.  Cloud
            # credential checks will still report google-auth as missing.
            account = cls._get_active_gcloud_account()
            legacy_account = None
        else:
            identity_type = cls._get_identity_type_from_credentials(credentials)
            account = cls._get_adc_principal(credentials, identity_type)
            legacy_account = cls._get_legacy_impersonation_source_principal(
                credentials)
        try:
            project_id = cls.get_project_id()
        except Exception as e:  # pylint: disable=broad-except
            with ux_utils.print_exception_no_traceback():
                raise exceptions.CloudUserIdentityError(
                    f'Failed to get GCP user identity with unknown '
                    f'exception.\n'
                    '  Reason: '
                    f'{common_utils.format_exception(e, use_bracket=True)}'
                ) from e
        identities = [[f'{account} [project_id={project_id}]']]
        if (legacy_account is not None and
                legacy_account.casefold() != account.casefold()):
            identities.append([f'{legacy_account} [project_id={project_id}]'])
        return identities

    @classmethod
    def get_active_user_identity_str(cls) -> Optional[str]:
        user_identity = cls.get_active_user_identity()
        if user_identity is None:
            return None
        return user_identity[0].replace('\n', '')

    def instance_type_exists(self, instance_type):
        return catalog.instance_type_exists(instance_type, 'gcp')

    def need_cleanup_after_preemption_or_failure(
            self, resources: 'resources.Resources') -> bool:
        """Whether a resource needs cleanup after preemption or failure."""
        # Spot TPU VMs require manual cleanup after preemption.
        # "If your Cloud TPU is preempted,
        # you must delete it and create a new one ..."
        # See: https://cloud.google.com/tpu/docs/preemptible#tpu-vm
        # On-demand TPU VMs are likely to require manual cleanup as well.

        return (gcp_utils.is_tpu_vm(resources) or
                _is_compute_tpu_instance_type(resources.instance_type))

    @classmethod
    def get_project_id(cls, dryrun: bool = False) -> str:
        if dryrun:
            return 'dryrun-project-id'
        config_project_id = skypilot_config.get_workspace_cloud('gcp').get(
            'project_id', None)
        if config_project_id:
            return config_project_id
        _, project_id = _get_default()
        if project_id is None:
            raise exceptions.CloudUserIdentityError(
                'Failed to get GCP project id. Please make sure you have '
                'run the following: gcloud init; '
                'gcloud auth application-default login')
        return project_id

    @staticmethod
    def _check_instance_type_accelerators_combination(
            resources: 'resources.Resources') -> None:
        resources = resources.assert_launchable()
        catalog.check_accelerator_attachable_to_host(resources.instance_type,
                                                     resources.accelerators,
                                                     resources.zone, 'gcp')

    @classmethod
    def check_disk_tier(
        cls,
        instance_type: Optional[str],  # pylint: disable=unused-argument
        disk_tier: Optional[resources_utils.DiskTier]  # pylint: disable=unused-argument
    ) -> Tuple[bool, str]:
        return True, ''

    @classmethod
    def check_disk_tier_enabled(cls, instance_type: Optional[str],
                                disk_tier: resources_utils.DiskTier) -> None:
        ok, msg = cls.check_disk_tier(instance_type, disk_tier)
        if not ok:
            with ux_utils.print_exception_no_traceback():
                raise exceptions.NotSupportedError(msg)

    @classmethod
    def _get_disk_type(
        cls,
        instance_type: Optional[str],
        disk_tier: Optional[resources_utils.DiskTier],
    ) -> str:

        def _propagate_disk_type(
            lowest: Optional[str] = None,
            highest: Optional[str] = None,
            # pylint: disable=redefined-builtin
            all: Optional[str] = None) -> None:
            if lowest is not None:
                tier2name[resources_utils.DiskTier.LOW] = lowest
            if highest is not None:
                tier2name[resources_utils.DiskTier.ULTRA] = highest
            if all is not None:
                for tier in tier2name:
                    tier2name[tier] = all

        tier = cls._translate_disk_tier(disk_tier)

        # Define the default mapping from disk tiers to disk types.
        tier2name = {
            resources_utils.DiskTier.ULTRA: 'pd-extreme',
            resources_utils.DiskTier.HIGH: 'pd-ssd',
            resources_utils.DiskTier.MEDIUM: 'pd-balanced',
            resources_utils.DiskTier.LOW: 'pd-standard',
        }

        # Remap series-specific disk types.
        # Reference: https://github.com/skypilot-org/skypilot/issues/4705
        assert instance_type is not None, (instance_type, disk_tier)
        series = instance_type.split('-')[0]

        # General handling of unsupported disk types
        if series in ['n1', 'a2', 'g2']:
            # These series don't support pd-extreme, use pd-ssd for ULTRA.
            _propagate_disk_type(
                highest=tier2name[resources_utils.DiskTier.HIGH])
        if series in ['a3', 'g2']:
            # These series don't support pd-standard, use pd-balanced for LOW.
            _propagate_disk_type(
                lowest=tier2name[resources_utils.DiskTier.MEDIUM])
        if instance_type.startswith('a3-ultragpu') or series in ('n4', 'a4',
                                                                 'g4'):
            # a3-ultragpu, n4, a4, and g4 instances only support
            # hyperdisk-balanced.
            _propagate_disk_type(all='hyperdisk-balanced')
        if series in ('ct6e', 'tpu7x'):
            # Compute Engine TPU VMs reject persistent disk boot disk types
            # such as pd-balanced.
            _propagate_disk_type(all='hyperdisk-balanced')

        # Series specific handling
        if series == 'n2':
            num_cpus = int(instance_type.split('-')[2])  # type: ignore
            if num_cpus < 64:
                # n2 series with less than 64 vCPUs doesn't support pd-extreme, use pd-ssd for ULTRA.
                _propagate_disk_type(
                    highest=tier2name[resources_utils.DiskTier.HIGH])
        elif series == 'a3':
            # LOW disk tier is already handled in general case, so in this branch
            # only the hyperdisk tier is addressed.
            tier2name[resources_utils.DiskTier.ULTRA] = 'hyperdisk-balanced'

        return tier2name[tier]

    @classmethod
    def _get_data_disk_type(
        cls,
        instance_type: Optional[str],
        disk_tier: Optional[resources_utils.DiskTier],
    ) -> str:

        tier = cls._translate_disk_tier(disk_tier)
        tier2name = volume_utils.get_data_disk_tier_mapping(instance_type)
        return tier2name[tier]

    @classmethod
    def _get_disk_specs(
            cls, instance_type: Optional[str],
            disk_tier: Optional[resources_utils.DiskTier]) -> Dict[str, Any]:
        specs: Dict[str, Any] = {
            'disk_tier': cls._get_disk_type(instance_type, disk_tier)
        }
        if (disk_tier == resources_utils.DiskTier.ULTRA and
                specs['disk_tier'] == 'pd-extreme'):
            # Only pd-extreme supports custom iops.
            # see https://cloud.google.com/compute/docs/disks#disk-types
            specs['disk_iops'] = constants.PD_EXTREME_IOPS
        return specs

    @classmethod
    def _get_volumes_specs(
        cls,
        region: 'clouds.Region',
        zones: Optional[List['clouds.Zone']],
        instance_type: Optional[str],
        volumes: Optional[List[Dict[str, Any]]],
        use_mig: bool,
        tpu_vm: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        if volumes is None:
            return [], {}

        project_id = cls.get_project_id()

        volume_utils.validate_instance_volumes(instance_type, volumes)

        volumes_specs: List[Dict[str, Any]] = []
        device_mount_points: Dict[str, str] = {}
        ssd_index = 0
        # TPU data disk index starts from 1, 0 is the boot disk
        tpu_disk_index = 1
        for i, volume in enumerate(volumes):
            volume_spec = {
                'device_name': f'sky-disk-{i}',
                'auto_delete': volume['auto_delete'],
            }
            if ('name' in volume and volume['storage_type']
                    == resources_utils.StorageType.NETWORK):
                volume_info = volume_utils.check_volume_name_exist_in_region(
                    project_id, region, use_mig, volume['name'])
                if volume_info is not None:
                    volume_utils.check_volume_zone_match(
                        volume['name'], zones, volume_info['available_zones'])
                    volume_spec['source'] = volume_info['selfLink']
                    volume_spec[
                        'attach_mode'] = volume_utils.translate_attach_mode(
                            volume['attach_mode'])
                    volume_spec['storage_type'] = constants.NETWORK_STORAGE_TYPE
                    volumes_specs.append(volume_spec)
                    device_name = f'{constants.DEVICE_NAME_PREFIX}sky-disk-{i}'
                    if tpu_vm:
                        # TPU VM does not support specifying the device name,
                        # so we use the default device name.
                        device_name = f'{constants.DEVICE_NAME_PREFIX}persistent-disk-{tpu_disk_index}'
                        tpu_disk_index += 1
                    device_mount_points[device_name] = volume['path']
                    continue
            if tpu_vm:
                # TODO(hailong): support creating block storage for TPU VM
                continue
            if volume['storage_type'] == resources_utils.StorageType.INSTANCE:
                device_name = f'{constants.INSTANCE_STORAGE_DEVICE_NAME_PREFIX}{ssd_index}'
                ssd_index += 1
                device_mount_points[device_name] = volume['path']

                if instance_type is not None and instance_type in constants.SSD_AUTO_ATTACH_MACHINE_TYPES:
                    # The instance storage will be attached automatically,
                    # so we skip the following steps.
                    continue

                volume_spec['disk_tier'] = constants.INSTANCE_STORAGE_DISK_TYPE
                volume_spec[
                    'interface_type'] = constants.INSTANCE_STORAGE_INTERFACE_TYPE
                volume_spec['storage_type'] = constants.INSTANCE_STORAGE_TYPE
                # Disk size of instance storage is fixed to 375GB
                volume_spec['disk_size'] = None
                volume_spec['auto_delete'] = True
            else:
                # TODO(hailong): this should be fixed when move the
                # disk creation out of the instance creation phase
                if not use_mig:
                    volume_spec['disk_name'] = volume['name']
                device_name = f'{constants.DEVICE_NAME_PREFIX}sky-disk-{i}'
                device_mount_points[device_name] = volume['path']

                volume_spec['storage_type'] = constants.NETWORK_STORAGE_TYPE
                if 'disk_size' in volume:
                    volume_spec['disk_size'] = volume['disk_size']
                else:
                    volume_spec['disk_size'] = constants.DEFAULT_DISK_SIZE
                disk_tier = cls.failover_disk_tier(instance_type,
                                                   volume['disk_tier'])
                volume_spec['disk_tier'] = cls._get_data_disk_type(
                    instance_type, disk_tier)
                if volume_spec['disk_tier'] == 'pd-extreme':
                    # Only pd-extreme supports custom iops.
                    # see https://cloud.google.com/compute/docs/disks#disk-types
                    volume_spec['disk_iops'] = constants.PD_EXTREME_IOPS
            volumes_specs.append(volume_spec)

        return volumes_specs, device_mount_points

    @classmethod
    def _label_filter_str(cls, tag_filters: Dict[str, str]) -> str:
        return ' '.join(f'labels.{k}={v}' for k, v in tag_filters.items())

    @classmethod
    def check_quota_available(cls, resources: 'resources.Resources') -> bool:
        """Check if GCP quota is available based on `resources`.

        GCP-specific implementation of check_quota_available. The function works by
        matching the `accelerator` to the a corresponding GCP keyword, and then using
        the GCP CLI commands to query for the specific quota (the `accelerator` as
        defined by `resources`).

        Returns:
            False if the quota is found to be zero, and True otherwise.
        Raises:
            CalledProcessError: error with the GCP CLI command.
        """

        if not resources.accelerators:
            # TODO(hriday): We currently only support checking quotas for GPUs.
            # For CPU-only instances, we need to try provisioning to check quotas.
            return True

        accelerator = list(resources.accelerators.keys())[0]
        use_spot = resources.use_spot
        region = resources.region
        managed_instance_group_config = (
            skypilot_config.get_effective_region_config(
                cloud='gcp',
                region=region,
                keys=('managed_instance_group',),
                default_value=None,
                override_configs=resources.cluster_config_overrides))
        if (managed_instance_group_config is not None and
                _is_managed_instance_group_eligible(resources)):
            # Flex-start VMs use DWS. GCP documents that these requests consume
            # preemptible quota once a project has requested preemptible quota;
            # projects that have never requested preemptible quota may consume
            # standard quota instead. Avoid incorrectly failing early on the
            # on-demand quota check and let the DWS provisioning request handle
            # quota/capacity.
            logger.warning(
                'Skipping GCP quota precheck for DWS/Flex-start. The DWS '
                'provisioning request will validate quota and capacity.')
            return True

        # pylint: disable=import-outside-toplevel
        from sky.catalog import gcp_catalog

        quota_code = gcp_catalog.get_quota_code(accelerator, use_spot)

        if quota_code is None:
            # Quota code not found in the catalog for the chosen instance_type, try provisioning anyway
            return True

        command = f'gcloud compute regions describe {region} |grep -B 1 "{quota_code}" | awk \'/limit/ {{print; exit}}\''
        try:
            proc = subprocess_utils.run(cmd=command,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)

        except subprocess.CalledProcessError as e:
            logger.warning(f'Quota check command failed with error: '
                           f'{e.stderr.decode()}')
            return True

        # Extract quota from output
        # Example output:  "- limit: 16.0"
        out = proc.stdout.decode()
        try:
            quota = int(float(out.split('limit:')[-1].strip()))
        except (ValueError, IndexError, AttributeError) as e:
            logger.warning('Parsing the subprocess output failed '
                           f'with error: {e}')
            return True

        if quota == 0:
            return False
        # Quota found to be greater than zero, try provisioning
        return True

    def get_reservations_available_resources(
        self,
        instance_type: str,
        region: str,
        zone: Optional[str],
        specific_reservations: Set[str],
    ) -> Dict[str, int]:
        del region  # Unused
        if zone is None:
            # For backward compatibility, the cluster in INIT state launched
            # before #2352 may not have zone information. In this case, we
            # return 0 for all reservations.
            return {reservation: 0 for reservation in specific_reservations}
        reservations = gcp_utils.list_reservations_for_instance_type_in_zone(
            instance_type, zone)

        return {
            r.name: r.available_resources
            for r in reservations
            if r.is_consumable(specific_reservations)
        }

    @classmethod
    def query_status(cls, name: str, tag_filters: Dict[str, str],
                     region: Optional[str], zone: Optional[str],
                     **kwargs) -> List['status_lib.ClusterStatus']:
        """Query the status of a cluster."""
        # TODO(suquark): deprecate this method
        assert False, 'This code path should not be used.'

    @classmethod
    def create_image_from_cluster(cls,
                                  cluster_name: resources_utils.ClusterName,
                                  region: Optional[str],
                                  zone: Optional[str]) -> str:
        del region  # unused
        assert zone is not None
        # TODO(zhwu): This assumes the cluster is created with the
        # `ray-cluster-name` tag, which is guaranteed by the current `ray`
        # backend. Once the `provision.query_instances` is implemented for GCP,
        # we should be able to get rid of this assumption.
        tag_filters = {'ray-cluster-name': cluster_name.name_on_cloud}
        label_filter_str = cls._label_filter_str(tag_filters)
        instance_name_cmd = ('gcloud compute instances list '
                             f'--filter="({label_filter_str})" '
                             '--format="json(name)"')
        returncode, stdout, stderr = subprocess_utils.run_with_retries(
            instance_name_cmd,
            retry_returncode=[255],
        )
        subprocess_utils.handle_returncode(
            returncode,
            instance_name_cmd,
            error_msg=
            f'Failed to get instance name for {cluster_name.display_name!r}',
            stderr=stderr,
            stream_logs=True)
        instance_names = json.loads(stdout)
        if len(instance_names) != 1:
            with ux_utils.print_exception_no_traceback():
                raise exceptions.NotSupportedError(
                    'Only support creating image from single '
                    f'instance, but got: {instance_names}')
        instance_name = instance_names[0]['name']

        image_name = f'skypilot-{cluster_name.display_name}-{int(time.time())}'
        create_image_cmd = (f'gcloud compute images create {image_name} '
                            f'--source-disk  {instance_name} '
                            f'--source-disk-zone {zone}')
        logger.debug(create_image_cmd)
        subprocess_utils.run_with_retries(
            create_image_cmd,
            retry_returncode=[255],
        )
        subprocess_utils.handle_returncode(
            returncode,
            create_image_cmd,
            error_msg=
            f'Failed to create image for {cluster_name.display_name!r}',
            stderr=stderr,
            stream_logs=True)

        image_uri_cmd = (f'gcloud compute images describe {image_name} '
                         '--format="get(selfLink)"')
        returncode, stdout, stderr = subprocess_utils.run_with_retries(
            image_uri_cmd,
            retry_returncode=[255],
        )

        subprocess_utils.handle_returncode(
            returncode,
            image_uri_cmd,
            error_msg=
            f'Failed to get image uri for {cluster_name.display_name!r}',
            stderr=stderr,
            stream_logs=True)

        image_uri = stdout.strip()
        image_id = image_uri.partition('projects/')[2]
        image_id = 'projects/' + image_id
        return image_id

    @classmethod
    def maybe_move_image(cls, image_id: str, source_region: str,
                         target_region: str, source_zone: Optional[str],
                         target_zone: Optional[str]) -> str:
        del source_region, target_region, source_zone, target_zone  # Unused.
        # GCP images are global, so no need to move.
        return image_id

    @classmethod
    def delete_image(cls, image_id: str, region: Optional[str]) -> None:
        del region  # Unused.
        image_name = image_id.rpartition('/')[2]
        delete_image_cmd = f'gcloud compute images delete {image_name} --quiet'
        returncode, _, stderr = subprocess_utils.run_with_retries(
            delete_image_cmd,
            retry_returncode=[255],
        )
        subprocess_utils.handle_returncode(
            returncode,
            delete_image_cmd,
            error_msg=f'Failed to delete image {image_name!r}',
            stderr=stderr,
            stream_logs=True)

    @classmethod
    def is_label_valid(cls, label_key: str,
                       label_value: str) -> Tuple[bool, Optional[str]]:
        key_regex = re.compile(r'^[a-z]([a-z0-9_-]{0,62})?$')
        value_regex = re.compile(r'^[a-z0-9_-]{0,63}$')
        key_valid = bool(key_regex.match(label_key))
        value_valid = bool(value_regex.match(label_value))
        error_msg = None
        condition_msg = ('can include lowercase alphanumeric characters, '
                         'dashes, and underscores, with a total length of 63 '
                         'characters or less.')
        if not key_valid:
            error_msg = (f'Invalid label key {label_key} for GCP. '
                         f'Key must start with a lowercase letter '
                         f'and {condition_msg}')
        if not value_valid:
            error_msg = (f'Invalid label value {label_value} for GCP. Value '
                         f'{condition_msg}')
        if not key_valid or not value_valid:
            return False, error_msg
        return True, None
