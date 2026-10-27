"""Unit tests for CloudVmRayBackend task configuration redaction and locking."""

import multiprocessing
import socket
import threading
import time
import types
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from sky import clouds
from sky import exceptions
from sky import resources
from sky import task
from sky.backends import backend_utils
from sky.backends import cloud_vm_ray_backend
from sky.backends.cloud_vm_ray_backend import CloudVmRayResourceHandle
from sky.backends.cloud_vm_ray_backend import SSHTunnelInfo
from sky.provision import common as provision_common
from sky.skylet import constants as skylet_constants
from sky.utils import locks
from sky.utils import status_lib


def test_command_length_local_shell_limit():
    # The local-shell check, still used by generated code that runs on the
    # cluster. Per-transport limits live on CommandRunner instead; see
    # tests/unit_tests/test_sky/utils/test_command_runner.py.
    command = "a'b" * 5000
    assert not backend_utils.is_command_length_over_limit(command)
    assert backend_utils.is_command_length_over_limit(command, quote_levels=4)


def test_non_slurm_cpu_demand_uses_ray_default():
    handle = MagicMock()
    handle.launched_resources = resources.Resources(cloud=clouds.Kubernetes())
    test_task = task.Task(resources=resources.Resources(cpus=1.5))

    demands = (cloud_vm_ray_backend.CloudVmRayBackend._get_task_demands_dict(
        handle, test_task))

    assert demands['CPU'] == backend_utils.DEFAULT_TASK_CPU_DEMAND


def test_slurm_controller_cpu_demand_uses_controller_default():
    handle = MagicMock()
    handle.launched_resources = resources.Resources(cloud=clouds.Slurm())
    test_task = task.Task(resources=resources.Resources(cpus=4))
    test_task.service_name = 'service'

    demands = (cloud_vm_ray_backend.CloudVmRayBackend._get_task_demands_dict(
        handle, test_task))

    assert (
        demands['CPU'] == backend_utils.constants.CONTROLLER_PROCESS_CPU_DEMAND)


def test_slurm_cpu_demand_uses_allocated_cpus():
    allocated = resources.Resources(cloud=clouds.Slurm(),
                                    instance_type='2CPU--2GB')
    handle = MagicMock()
    handle.launched_resources = allocated
    test_task = task.Task(resources=resources.Resources(cpus=0.25))
    test_task.best_resources = allocated

    demands = (cloud_vm_ray_backend.CloudVmRayBackend._get_task_demands_dict(
        handle, test_task))

    assert demands['CPU'] == 2.0


def test_slurm_gpu_cpu_demand_uses_allocated_cpus():
    allocated = resources.Resources(cloud=clouds.Slurm(),
                                    instance_type='4CPU--16GB--H200:1')
    handle = MagicMock()
    handle.launched_resources = allocated
    test_task = task.Task(resources=resources.Resources(
        accelerators={'H200': 1}))
    test_task.best_resources = allocated

    demands = (cloud_vm_ray_backend.CloudVmRayBackend._get_task_demands_dict(
        handle, test_task))

    assert demands['CPU'] == 4.0


def test_slurm_exec_cpu_demand_uses_cluster_allocation():
    handle = MagicMock()
    handle.launched_resources = resources.Resources(cloud=clouds.Slurm(),
                                                    instance_type='8CPU--32GB')
    test_task = task.Task(resources=resources.Resources())

    demands = (cloud_vm_ray_backend.CloudVmRayBackend._get_task_demands_dict(
        handle, test_task))

    assert demands['CPU'] == 8.0


def test_pause_on_slurm_ssh_failure():
    handle = MagicMock()
    handle.launched_resources.cloud = clouds.Slurm()

    with pytest.raises(exceptions.ExecutionPausedError):
        cloud_vm_ray_backend._pause_on_slurm_ssh_failure(handle, 255)

    cloud_vm_ray_backend._pause_on_slurm_ssh_failure(handle, 1)


class TestCloudVmRayBackendTaskRedaction:
    """Tests for CloudVmRayBackend usage of redacted task configs."""

    def test_cloud_vm_ray_backend_redaction_usage_pattern(self):
        """Test the exact usage pattern from the CloudVmRayBackend code."""
        # Create a task with sensitive secret variables and regular environment variables
        test_task = task.Task(run='echo hello',
                              envs={
                                  'DEBUG': 'true',
                                  'PORT': '8080'
                              },
                              secrets={
                                  'API_KEY': 'sk-very-secret-key-123',
                                  'DATABASE_PASSWORD': 'super-secret-password',
                                  'JWT_SECRET': 'jwt-signing-secret-456'
                              })

        # Test the exact call pattern used in cloud_vm_ray_backend.py
        task_config = test_task.to_yaml_config(use_user_specified_yaml=True)

        # Verify that environment variables are NOT redacted
        assert 'envs' in task_config
        assert task_config['envs']['DEBUG'] == 'true'
        assert task_config['envs']['PORT'] == '8080'

        # Verify that secrets ARE redacted
        assert 'secrets' in task_config
        assert task_config['secrets']['API_KEY'] == '<redacted>'
        assert task_config['secrets']['DATABASE_PASSWORD'] == '<redacted>'
        assert task_config['secrets']['JWT_SECRET'] == '<redacted>'

        # Verify other task fields are preserved
        assert task_config['run'] == 'echo hello'

    def test_backend_task_config_without_secrets(self):
        """Test task config generation when no secrets are present."""
        test_task = task.Task(run='python train.py',
                              envs={'PYTHONPATH': '/app'})

        task_config = test_task.to_yaml_config(use_user_specified_yaml=True)

        # Environment variables should be preserved
        assert task_config['envs']['PYTHONPATH'] == '/app'

        # No secrets field should be present
        assert 'secrets' not in task_config or not task_config.get('secrets')

    def test_backend_task_config_empty_secrets(self):
        """Test task config generation with empty secrets dict."""
        test_task = task.Task(run='python train.py',
                              envs={'PYTHONPATH': '/app'},
                              secrets={})

        task_config = test_task.to_yaml_config(use_user_specified_yaml=True)

        # Environment variables should be preserved
        assert task_config['envs']['PYTHONPATH'] == '/app'

        # Empty secrets should not appear in config
        assert 'secrets' not in task_config

    def test_backend_redaction_redacts_all_values(self):
        """Test that all secret values (including non-string) are redacted."""
        test_task = task.Task(run='echo hello',
                              secrets={
                                  'STRING_SECRET': 'actual-secret',
                                  'NUMERIC_PORT': 5432,
                                  'BOOLEAN_FLAG': True,
                                  'NULL_VALUE': None
                              })

        task_config = test_task.to_yaml_config(use_user_specified_yaml=True)

        # String values should be redacted
        assert task_config['secrets']['STRING_SECRET'] == '<redacted>'

        # All values should be redacted (including non-string values)
        assert task_config['secrets']['NUMERIC_PORT'] == '<redacted>'
        assert task_config['secrets']['BOOLEAN_FLAG'] == '<redacted>'
        assert task_config['secrets']['NULL_VALUE'] == '<redacted>'

    def test_backend_supports_both_redaction_modes(self):
        """Test that backend can use both redacted and non-redacted configs."""
        test_task = task.Task(run='echo hello',
                              secrets={'API_KEY': 'secret-key-123'})

        # Test redacted mode (for logging/display)
        redacted_config = test_task.to_yaml_config(use_user_specified_yaml=True)
        assert redacted_config['secrets']['API_KEY'] == '<redacted>'

        # Test non-redacted mode (for execution)
        full_config = test_task.to_yaml_config(use_user_specified_yaml=False)
        assert full_config['secrets']['API_KEY'] == 'secret-key-123'

    def test_backend_mixed_envs_and_secrets(self):
        """Test backend behavior with both envs and secrets containing sensitive data."""
        test_task = task.Task(
            run='echo hello',
            envs={
                'PUBLIC_API_URL': 'https://api.example.com',
                'DEBUG_MODE': 'true',
                'ENVIRONMENT': 'production'
            },
            secrets={
                'PRIVATE_API_KEY': 'sk-secret-key-123',
                'DATABASE_URL': 'postgresql://user:pass@host:5432/db',
                'OAUTH_CLIENT_SECRET': 'oauth-secret-456'
            })

        task_config = test_task.to_yaml_config(use_user_specified_yaml=True)

        # All environment variables should remain visible
        assert task_config['envs'][
            'PUBLIC_API_URL'] == 'https://api.example.com'
        assert task_config['envs']['DEBUG_MODE'] == 'true'
        assert task_config['envs']['ENVIRONMENT'] == 'production'

        # All secrets should be redacted
        assert task_config['secrets']['PRIVATE_API_KEY'] == '<redacted>'
        assert task_config['secrets']['DATABASE_URL'] == '<redacted>'
        assert task_config['secrets']['OAUTH_CLIENT_SECRET'] == '<redacted>'

    def test_backend_config_serialization_safety(self):
        """Test that redacted configs are safe for serialization/logging."""
        import json

        import yaml

        test_task = task.Task(
            run='echo hello',
            envs={'PUBLIC_VAR': 'public-value'},
            secrets={'PRIVATE_KEY': 'very-sensitive-key-data'})

        redacted_config = test_task.to_yaml_config(use_user_specified_yaml=True)

        # Should be serializable to JSON
        json_str = json.dumps(redacted_config)
        assert 'very-sensitive-key-data' not in json_str
        assert '<redacted>' in json_str

        # Should be serializable to YAML
        yaml_str = yaml.dump(redacted_config)
        assert 'very-sensitive-key-data' not in yaml_str
        assert '<redacted>' in yaml_str

        # Public values should still be present
        assert 'public-value' in yaml_str

    def test_redacted_config_contains_no_sensitive_data(self):
        """Test that redacted task config doesn't contain sensitive secret data."""
        # Create a task with sensitive secret variables and regular environment variables
        test_task = task.Task(run='echo hello',
                              envs={
                                  'DEBUG': 'true',
                                  'PORT': 8080,
                                  'PUBLIC_VAR': 'public-value'
                              },
                              secrets={
                                  'API_KEY': 'secret-api-key-123',
                                  'DATABASE_PASSWORD': 'super-secret-password',
                                  'AWS_SECRET_ACCESS_KEY': 'aws-secret-key',
                                  'STRIPE_SECRET_KEY': 'sk_live_sensitive_key',
                                  'JWT_SECRET': 'jwt-signing-secret',
                              })

        # Get the redacted config as the backend would
        redacted_config = test_task.to_yaml_config(use_user_specified_yaml=True)

        # Verify sensitive string values in secrets are redacted
        assert redacted_config['secrets']['API_KEY'] == '<redacted>'
        assert redacted_config['secrets']['DATABASE_PASSWORD'] == '<redacted>'
        assert redacted_config['secrets'][
            'AWS_SECRET_ACCESS_KEY'] == '<redacted>'
        assert redacted_config['secrets']['STRIPE_SECRET_KEY'] == '<redacted>'
        assert redacted_config['secrets']['JWT_SECRET'] == '<redacted>'

        # Verify envs are NOT redacted (preserved as-is)
        assert redacted_config['envs']['DEBUG'] == 'true'
        assert redacted_config['envs']['PORT'] == 8080
        assert redacted_config['envs']['PUBLIC_VAR'] == 'public-value'

        # Ensure no sensitive data appears anywhere in the config
        config_str = str(redacted_config)
        assert 'secret-api-key-123' not in config_str
        assert 'super-secret-password' not in config_str
        assert 'aws-secret-key' not in config_str
        assert 'sk_live_sensitive_key' not in config_str
        assert 'jwt-signing-secret' not in config_str

        # But public values should still be present
        assert 'public-value' in config_str

    def test_non_redacted_config_contains_actual_values(self):
        """Test that non-redacted config contains actual secret values."""
        # Create a task with environment variables and secrets
        test_task = task.Task(run='echo hello',
                              envs={
                                  'DEBUG': 'true',
                                  'PORT': 8080
                              },
                              secrets={
                                  'API_KEY': 'actual-api-key',
                                  'JWT_SECRET': 'actual-jwt-secret'
                              })

        # Get the non-redacted config
        non_redacted_config = test_task.to_yaml_config(
            use_user_specified_yaml=False)

        # Verify actual values are present in both envs and secrets
        assert non_redacted_config['envs']['DEBUG'] == 'true'
        assert non_redacted_config['envs']['PORT'] == 8080
        assert non_redacted_config['secrets']['API_KEY'] == 'actual-api-key'
        assert non_redacted_config['secrets'][
            'JWT_SECRET'] == 'actual-jwt-secret'

        # Also test default behavior (should NOT redact secrets by default)
        default_config = test_task.to_yaml_config()
        assert default_config['envs'] == non_redacted_config[
            'envs']  # envs same
        assert default_config['secrets'][
            'API_KEY'] == 'actual-api-key'  # secrets not redacted by default

    def test_backend_redaction_with_no_secrets(self):
        """Test backend behavior when task has no secret variables."""
        # Create a task with only environment variables, no secrets
        test_task = task.Task(run='echo hello', envs={'DEBUG': 'true'})

        # Get redacted config
        redacted_config = test_task.to_yaml_config(use_user_specified_yaml=True)

        # Should not have secrets key at all
        assert 'secrets' not in redacted_config

        # Should have envs key with actual values (not redacted)
        assert 'envs' in redacted_config
        assert redacted_config['envs']['DEBUG'] == 'true'

        # Should still have other task properties
        assert redacted_config['run'] == 'echo hello'

    def test_backend_redaction_preserves_task_structure(self):
        """Test that redaction preserves all non-secret task configuration."""
        from sky import resources

        # Create a comprehensive task
        test_task = task.Task(run='python train.py',
                              envs={
                                  'DEBUG': 'true',
                                  'PORT': 8080
                              },
                              secrets={
                                  'API_KEY': 'secret-value',
                                  'DB_PASSWORD': 'secret-password'
                              },
                              workdir='/app',
                              name='training-task')
        # Set resources using the proper method
        test_task.set_resources(resources.Resources(cpus=4, memory=8))

        # Get both configs
        original_config = test_task.to_yaml_config(
            use_user_specified_yaml=False)
        redacted_config = test_task.to_yaml_config(use_user_specified_yaml=True)

        # All non-secret fields should be identical
        for key in original_config:
            if key != 'secrets':
                assert original_config[key] == redacted_config[key]

        # Envs should be identical (not redacted)
        assert original_config['envs'] == redacted_config['envs']
        assert redacted_config['envs']['DEBUG'] == 'true'
        assert redacted_config['envs']['PORT'] == 8080

        # Secret handling should be different
        assert original_config['secrets']['API_KEY'] == 'secret-value'
        assert redacted_config['secrets']['API_KEY'] == '<redacted>'
        assert original_config['secrets']['DB_PASSWORD'] == 'secret-password'
        assert redacted_config['secrets']['DB_PASSWORD'] == '<redacted>'


class TestCloudVmRayBackendGetGrpcChannel:
    """Tests for CloudVmRayBackend get_grpc_channel."""
    MOCK_HANDLE_KWARGS = {
        'cluster_name': 'test-cluster',
        'cluster_name_on_cloud': 'test-cluster-abc',
        'cluster_yaml': None,
        'launched_nodes': 1,
        'launched_resources': MagicMock(),
    }

    INITIAL_TUNNEL_PORT = 10000
    INITIAL_TUNNEL_PID = 12345
    PROCESS_JOIN_TIMEOUT_SECONDS = 30

    def _simulate_process_get_grpc_channel(self, queue, tunnel_creation_count,
                                           tunnel_port, tunnel_pid,
                                           socket_connect_side_effect):
        """Simulate a process calling get_grpc_channel.

        This test mocks:
        - _get_skylet_ssh_tunnel: To avoid making an actual DB query
        - _open_and_update_skylet_tunnel: To avoid actually opening an SSH tunnel
        - grpc.insecure_channel: To just return the address instead of a Channel object
        - socket.socket.connect: To avoid actually connecting to the tunnel

        This test does not mock:
        - lock.acquire
        """
        try:
            # Different processes have different handle instances.
            handle = CloudVmRayResourceHandle(**self.MOCK_HANDLE_KWARGS)

            def mock_get_tunnel_side_effect():
                # Return None if the tunnel is not created yet.
                if tunnel_port.value == -1 or tunnel_pid.value == -1:
                    return None
                return SSHTunnelInfo(port=tunnel_port.value,
                                     pid=tunnel_pid.value)

            def mock_open_tunnel():
                # Simulate time taken to create tunnel.
                time.sleep(2)
                with tunnel_creation_count.get_lock():
                    tunnel_creation_count.value += 1
                    created = tunnel_creation_count.value
                with tunnel_port.get_lock(), tunnel_pid.get_lock():
                    # First creation -> 10000/12345; second -> 10001/12346; and so on.
                    tunnel_port.value = self.INITIAL_TUNNEL_PORT + (created - 1)
                    tunnel_pid.value = self.INITIAL_TUNNEL_PID + (created - 1)
                    return SSHTunnelInfo(port=tunnel_port.value,
                                         pid=tunnel_pid.value)

            with patch.object(handle, '_get_skylet_ssh_tunnel', side_effect=mock_get_tunnel_side_effect), \
                patch.object(handle, '_open_and_update_skylet_tunnel', side_effect=mock_open_tunnel), \
                patch('grpc.insecure_channel', side_effect=lambda addr, options: addr), \
                patch('socket.socket') as mock_socket:

                mock_socket.return_value.__enter__.return_value.connect.side_effect = socket_connect_side_effect

                res = handle.get_grpc_channel()
                assert res is not None
                queue.put(res)

        except Exception as e:
            import traceback
            error_msg = f"Error: {e}\nTraceback: {traceback.format_exc()}"
            queue.put(error_msg)

    def _socket_connect_side_effect(self, addr):
        _, port = addr
        # Force an error on the original port to test the retry logic.
        if port == self.INITIAL_TUNNEL_PORT:
            raise socket.error("Connection error")
        return None

    @pytest.mark.parametrize('has_stale_tunnel', [False, True])
    @pytest.mark.parametrize('concurrent_tunnel_created', [False, True])
    def test_get_grpc_channel_rechecks_tunnel_after_acquiring_lock(
            self, has_stale_tunnel, concurrent_tunnel_created):
        """Reuse a tunnel created after the initial read but before locking."""
        handle = CloudVmRayResourceHandle(**self.MOCK_HANDLE_KWARGS)
        stale_tunnel = SSHTunnelInfo(port=self.INITIAL_TUNNEL_PORT,
                                     pid=self.INITIAL_TUNNEL_PID)
        healthy_tunnel = SSHTunnelInfo(port=self.INITIAL_TUNNEL_PORT + 1,
                                       pid=self.INITIAL_TUNNEL_PID + 1)
        tunnel_state = {'tunnel': stale_tunnel if has_stale_tunnel else None}

        def acquire_lock():
            # Another process finishes opening a tunnel just before this
            # process acquires the exclusive lock.
            if concurrent_tunnel_created:
                tunnel_state['tunnel'] = healthy_tunnel

        exclusive_lock = MagicMock()
        exclusive_lock.acquire.return_value.__enter__.side_effect = acquire_lock
        with patch.object(handle, '_get_skylet_ssh_tunnel',
                          side_effect=lambda: tunnel_state['tunnel']), \
                patch.object(handle, '_open_and_update_skylet_tunnel',
                             return_value=healthy_tunnel) as open_tunnel, \
                patch.object(locks, 'get_lock', return_value=exclusive_lock), \
                patch('grpc.insecure_channel') as channel, \
                patch('socket.socket') as mock_socket:
            mock_socket.return_value.__enter__.return_value.connect.side_effect = (
                self._socket_connect_side_effect)

            assert handle.get_grpc_channel() == channel.return_value
            assert channel.call_args.args[
                0] == f'localhost:{healthy_tunnel.port}'
            if concurrent_tunnel_created:
                open_tunnel.assert_not_called()
            else:
                open_tunnel.assert_called_once_with()

    def test_get_grpc_channel_multiprocess_race_condition(self):
        """Test get_grpc_channel with multiple processes racing for tunnel creation."""
        tunnel_creation_count = multiprocessing.Value('i', 0)
        tunnel_port = multiprocessing.Value('i', -1)
        tunnel_pid = multiprocessing.Value('i', -1)

        num_processes = 5
        processes = []
        queue = multiprocessing.Queue()
        for _ in range(num_processes):
            p = multiprocessing.Process(
                target=self._simulate_process_get_grpc_channel,
                args=(queue, tunnel_creation_count, tunnel_port, tunnel_pid,
                      None))
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=self.PROCESS_JOIN_TIMEOUT_SECONDS)
            if p.is_alive():
                p.terminate()
                p.join()

        results = []
        while not queue.empty():
            results.append(queue.get())
        assert len(
            results
        ) == num_processes, f"Expected {num_processes} results, got {len(results)}"
        # All processes should get the same channel (localhost:10000).
        for item in results:
            assert item == f'localhost:{self.INITIAL_TUNNEL_PORT}', f"Failed: {item}"

        assert tunnel_creation_count.value == 1, f"Expected tunnel to be created exactly once, but was created {tunnel_creation_count.value} times"

        # Try again, this tests the case where the tunnel is already created.
        # This time, tunnel.port will be 10000, but the check should fail,
        # as our _socket_connect_side_effect will raise an error. So we
        # should invoke _open_and_update_skylet_tunnel again,
        # this time returning another port.
        for _ in range(num_processes):
            p = multiprocessing.Process(
                target=self._simulate_process_get_grpc_channel,
                args=(queue, tunnel_creation_count, tunnel_port, tunnel_pid,
                      self._socket_connect_side_effect))
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=self.PROCESS_JOIN_TIMEOUT_SECONDS)
            if p.is_alive():
                p.terminate()
                p.join()

        results = []
        while not queue.empty():
            results.append(queue.get())
        assert len(
            results
        ) == num_processes, f"Expected {num_processes} results, got {len(results)}"

        # All processes should get the same channel (localhost:10001).
        for i in range(num_processes):
            assert results[
                i] == f'localhost:{self.INITIAL_TUNNEL_PORT + 1}', f"Process {i} failed: {results[i]}"

        assert tunnel_creation_count.value == 2, f"Expected tunnel to be created exactly once, but was created {tunnel_creation_count.value} times"

    def test_setup_num_gpus(self, monkeypatch):
        """Test setup num GPUs."""
        test_task = task.Task(resources=resources.Resources(
            accelerators={'A100': 8}))
        monkeypatch.setattr(CloudVmRayResourceHandle, '__init__',
                            lambda self, *args, **kwargs: None)
        backend = cloud_vm_ray_backend.CloudVmRayBackend()
        assert backend._get_num_gpus(test_task) == 8


class TestTpuFlexStartMultiZoneProvisioning:
    """Tests parallel TPU MIG DWS Flex-start zone provisioning."""

    @staticmethod
    def _provisioner():
        provisioner = cloud_vm_ray_backend.RetryingVmProvisioner.__new__(
            cloud_vm_ray_backend.RetryingVmProvisioner)
        provisioner._blocked_resources = set()
        provisioner.log_dir = '/tmp/sky-test-flex-start'
        provisioner._dag = MagicMock()
        provisioner._optimize_target = MagicMock()
        provisioner._requested_features = set()
        provisioner._local_wheel_path = MagicMock()
        provisioner._wheel_hash = 'wheel-hash'
        provisioner._is_managed = False
        provisioner._extra_launch_context = {}
        provisioner._is_launched_by_jobs_controller = False
        return provisioner

    @staticmethod
    def _resources(region=None, zone=None):
        return resources.Resources(cloud=clouds.GCP(),
                                   instance_type='ct6e-standard-4t',
                                   region=region,
                                   zone=zone,
                                   _cluster_config_overrides={
                                       'gcp': {
                                           'managed_instance_group': {
                                               'run_duration': 3600,
                                               'accelerator_topology': '4x4',
                                           }
                                       }
                                   })

    def _candidates(self, provisioner, launch_task, selected):
        regions = [
            clouds.Region('us-east5').set_zones([
                clouds.Zone('us-east5-a'),
                clouds.Zone('us-east5-b'),
            ]),
            clouds.Region('us-central1').set_zones(
                [clouds.Zone('us-central1-a')]),
            clouds.Region('europe-west4').set_zones(
                [clouds.Zone('europe-west4-a')]),
        ]
        with patch.object(resources.Resources,
                          'get_valid_regions_for_launchable',
                          return_value=regions), patch.object(
                              cloud_vm_ray_backend.skypilot_config,
                              'get_effective_region_config',
                              return_value={'run_duration': 3600}):
            candidates = provisioner._get_tpu_flex_start_candidates(
                launch_task, selected)
        return [(candidate.region, candidate.zone) for candidate in candidates]

    def test_unpinned_request_races_every_flex_start_zone(self):
        provisioner = self._provisioner()
        launch_task = task.Task(resources=self._resources(), num_nodes=4)
        selected = self._resources(region='us-east5', zone='us-east5-b')

        assert self._candidates(provisioner, launch_task, selected) == [
            ('us-east5', 'us-east5-b'),
            ('us-east5', 'us-east5-a'),
            ('us-central1', 'us-central1-a'),
            ('europe-west4', 'europe-west4-a'),
        ]

    def test_region_pinned_request_races_that_regions_zones(self):
        provisioner = self._provisioner()
        launch_task = task.Task(resources=self._resources(region='us-east5'),
                                num_nodes=4)
        selected = self._resources(region='us-east5', zone='us-east5-a')

        assert self._candidates(provisioner, launch_task, selected) == [
            ('us-east5', 'us-east5-a'),
            ('us-east5', 'us-east5-b'),
        ]

    def test_zone_pinned_request_does_not_race(self):
        provisioner = self._provisioner()
        launch_task = task.Task(resources=self._resources(region='us-east5',
                                                          zone='us-east5-a'),
                                num_nodes=4)
        selected = self._resources(region='us-east5', zone='us-east5-a')

        assert self._candidates(provisioner, launch_task, selected) == []

    def test_first_zone_wins_and_late_success_is_cleaned_up(self):
        provisioner = self._provisioner()
        requested = self._resources()
        launch_task = task.Task(resources=requested, num_nodes=4)
        candidates = [
            requested.copy(region='us-east5', zone='us-east5-a'),
            requested.copy(region='us-east5', zone='us-east5-b'),
        ]
        started = threading.Barrier(len(candidates))
        cancelled_zones = []
        configs = {}
        stored_yamls = {}

        def retry_zones(self, candidate, *args, cancel_event, **kwargs):
            del self, args
            handle = MagicMock()
            handle.cluster_name = kwargs['cluster_name']
            handle.cluster_name_on_cloud = name_on_cloud(kwargs['cluster_name'])
            handle.launched_resources = candidate
            handle.cluster_yaml = f'/tmp/{kwargs["cluster_name"]}.yml'
            config = {'handle': handle, 'provisioning_skipped': False}
            configs[candidate.zone] = config
            kwargs['on_cluster_config_created'](handle)
            started.wait()
            if candidate.zone == 'us-east5-a':
                return config
            assert cancel_event.wait(timeout=2)
            cancelled_zones.append(candidate.zone)
            return config

        def name_on_cloud(name):
            return cloud_vm_ray_backend.common_utils.make_cluster_name_on_cloud(
                name, max_length=clouds.GCP.max_cluster_name_length())

        def set_cluster_yaml(name, yaml_str):
            stored_yamls[name] = cloud_vm_ray_backend.yaml_utils.safe_load(
                yaml_str)

        with patch.object(
                provisioner,
                '_get_tpu_flex_start_candidates',
                return_value=candidates), patch.object(
                    cloud_vm_ray_backend.RetryingVmProvisioner,
                    '_retry_zones',
                    autospec=True,
                    side_effect=retry_zones), patch.object(
                        provisioner,
                        '_cleanup_tpu_flex_start_candidate') as cleanup, \
             patch.object(
                 provisioner,
                 '_record_cluster_provisioning_state') as record_state, \
             patch.object(cloud_vm_ray_backend.global_user_state,
                          'add_cluster_event'), \
             patch.object(cloud_vm_ray_backend.global_user_state,
                          'get_cluster_yaml_dict',
                          side_effect=lambda path: {'provider': {}}), \
             patch.object(cloud_vm_ray_backend.global_user_state,
                          'set_cluster_yaml',
                          side_effect=set_cluster_yaml):
            result = provisioner._race_tpu_flex_start_zones(
                launch_task, candidates[0], 4, False, 'test', ['user'])

        assert result is not None
        assert result['handle'].cluster_name == 'test'
        assert cancelled_zones == ['us-east5-b']
        cleanup.assert_called_once_with(configs['us-east5-b'])
        assert record_state.call_count == 2
        # Every candidate's YAML names all racing MIGs before submitting.
        expected_members = [{
            'zone': 'us-east5-a',
            'cluster_name_on_cloud': name_on_cloud('test-mr-us-east5-a'),
        }, {
            'zone': 'us-east5-b',
            'cluster_name_on_cloud': name_on_cloud('test-mr-us-east5-b'),
        }]
        assert set(stored_yamls) == {'test-mr-us-east5-a', 'test-mr-us-east5-b'}
        for config in stored_yamls.values():
            assert config['provider'][
                cloud_vm_ray_backend._TPU_FLEX_START_RACE_MEMBERS_KEY] == (
                    expected_members)

    def test_all_zones_fail_and_are_blocked(self):
        provisioner = self._provisioner()
        requested = self._resources()
        launch_task = task.Task(resources=requested, num_nodes=4)
        candidates = [
            requested.copy(region='us-east5', zone='us-east5-a'),
            requested.copy(region='europe-west4', zone='europe-west4-a'),
        ]

        with patch.object(provisioner,
                          '_get_tpu_flex_start_candidates',
                          return_value=candidates), patch.object(
                              cloud_vm_ray_backend.RetryingVmProvisioner,
                              '_retry_zones',
                              autospec=True,
                              side_effect=exceptions.ResourcesUnavailableError(
                                  'no capacity')), patch.object(
                                      provisioner,
                                      '_record_cluster_provisioning_state'):
            with pytest.raises(exceptions.ResourcesUnavailableError,
                               match='All parallel TPU Flex-start'):
                provisioner._race_tpu_flex_start_zones(launch_task,
                                                       candidates[0], 4, False,
                                                       'test', ['user'])

        assert provisioner._blocked_resources == set(candidates)

    def test_spot_tpu_mig_is_not_treated_as_flex_start(self):
        provisioner = self._provisioner()
        requested = self._resources().copy(use_spot=True)
        launch_task = task.Task(resources=requested, num_nodes=4)
        selected = requested.copy(region='us-east5', zone='us-east5-a')

        assert provisioner._get_tpu_flex_start_candidates(
            launch_task, selected) == []

    def test_race_member_cleanup_targets_other_zones(self):
        provider_config = {
            'project_id': 'proj',
            'region': 'us-east5',
            'availability_zone': 'us-east5-a',
            '_is_tpu_mig': True,
            cloud_vm_ray_backend._TPU_FLEX_START_RACE_MEMBERS_KEY: [{
                'zone': 'us-east5-a',
                'cluster_name_on_cloud': 'own',
            }, {
                'zone': 'europe-west4-a',
                'cluster_name_on_cloud': 'sibling',
            }],
        }

        with patch.object(cloud_vm_ray_backend.provision_lib,
                          'cleanup_cluster_resources') as cleanup:
            cloud_vm_ray_backend._cleanup_tpu_flex_start_race_members(
                'GCP', 'own', provider_config)

        cleanup.assert_called_once_with(
            'GCP', 'sibling', {
                'project_id': 'proj',
                'region': 'europe-west4',
                'availability_zone': 'europe-west4-a',
                '_is_tpu_mig': True,
            })

    @pytest.mark.parametrize('failover', [False, True])
    def test_teardown_sweeps_race_members_only_outside_failover(self, failover):
        backend = cloud_vm_ray_backend.CloudVmRayBackend()
        handle = MagicMock()
        handle.cluster_name = 'test'
        handle.cluster_name_on_cloud = 'test-mr-us-east5-a-cloud'
        handle.cluster_yaml = '/tmp/test-mr-us-east5-a.yml'
        handle.launched_resources.is_image_managed = False
        launchable = MagicMock()
        # Mock rejects reading attributes named assert_*, so set it directly.
        handle.launched_resources.assert_launchable = MagicMock(
            return_value=launchable)
        cloud = launchable.cloud
        provider_config = {'availability_zone': 'us-east5-a'}

        with patch.object(cloud_vm_ray_backend, 'provision_lib') as provision, \
             patch.object(cloud_vm_ray_backend, 'global_user_state') as state, \
             patch.object(cloud_vm_ray_backend, 'metadata_utils'), \
             patch.object(cloud_vm_ray_backend, 'cluster_utils'), \
             patch.object(backend, 'remove_cluster_config'), \
             patch.object(cloud_vm_ray_backend,
                          '_cleanup_tpu_flex_start_race_members') as sweep:
            state.get_cluster_yaml_dict.return_value = {
                'provider': provider_config
            }
            provision.query_instances.return_value = {}
            backend.post_teardown_cleanup(handle,
                                          terminate=True,
                                          remove_from_db=False,
                                          failover=failover)

        provision.cleanup_cluster_resources.assert_called_once_with(
            repr(cloud), 'test-mr-us-east5-a-cloud', provider_config)
        if failover:
            sweep.assert_not_called()
        else:
            sweep.assert_called_once_with(repr(cloud),
                                          'test-mr-us-east5-a-cloud',
                                          provider_config)


class TestIsMessageTooLong:
    """Tests for _is_message_too_long function."""

    @pytest.mark.parametrize(
        'returncode,message,expected',
        [
            # Valid matches with correct returncode
            (255, 'too long', True),
            (255, 'Argument list too long', True),
            (1, 'request-uri too large', True),
            (1, '414 Request-URI Too Large', True),
            (1, 'request header fields too large', True),
            (1, '431 Request Header Fields Too Large', True),
            # CloudFlare 400 Bad Request patterns
            (1, '400 bad request', True),
            (1, '400 Bad request', True),
            (1, '400 Bad Request', True),
            (1,
             'error: unable to upgrade connection: <html><body><h1>400 Bad request</h1>',
             True),
            # Signatures are matched as bare substrings against the whole
            # setup log, which is user output, so generic network-failure text
            # must not be in the table: a user script printing it and exiting 1
            # would have its setup re-run from the top.
            (1, 'read tcp 10.0.0.1:443: connection reset by peer', False),
            (1, 'gzip: unexpected EOF', False),
            (1, 'Error from server: ', False),
            # Case insensitivity
            (255, 'TOO LONG', True),
            (1, 'REQUEST HEADER FIELDS TOO LARGE', True),
            (1, '400 BAD REQUEST', True),
            # Wrong returncode
            (1, 'too long', False),
            (255, 'request-uri too large', False),
            (127, 'too long', False),
            (255, '400 bad request', False),
            # Wrong message
            (255, 'command not found', False),
            (1, 'some other error', False),
            (1, 'unable to upgrade connection', False),
            # Empty output
            (255, '', False),
        ])
    def test_detection_with_output(self, returncode, message, expected):
        """Test message detection with various returncode/message combinations."""
        assert cloud_vm_ray_backend._is_message_too_long(
            returncode, output=message) == expected

    def test_detection_with_file_path(self, tmp_path):
        """Test detection when reading from file."""
        log_file = tmp_path / "test.log"
        log_file.write_text("Error: command too long")
        assert cloud_vm_ray_backend._is_message_too_long(
            255, file_path=str(log_file))

        log_file.write_text("431 Request Header Fields Too Large")
        assert cloud_vm_ray_backend._is_message_too_long(
            1, file_path=str(log_file))

    def test_file_read_error_returns_true(self, tmp_path):
        """Test that file read errors return True for safety."""
        # Non-existent file
        assert cloud_vm_ray_backend._is_message_too_long(
            255, file_path="/nonexistent/file.log")

        # Unreadable file
        log_file = tmp_path / "unreadable.log"
        log_file.write_text("content")
        log_file.chmod(0o000)
        try:
            assert cloud_vm_ray_backend._is_message_too_long(
                255, file_path=str(log_file))
        finally:
            log_file.chmod(0o644)

    def test_requires_either_output_or_file_path(self):
        """Test that function requires either output or file_path."""
        with pytest.raises(AssertionError):
            cloud_vm_ray_backend._is_message_too_long(255)
        with pytest.raises(AssertionError):
            cloud_vm_ray_backend._is_message_too_long(255,
                                                      output="test",
                                                      file_path="/tmp/test")

    def test_partial_match_in_long_output(self):
        """Test that partial matches in longer messages are detected."""
        long_output = """Error executing command on remote server:
        bash: /usr/bin/ssh: Argument list too long
        Failed to run setup script"""
        assert cloud_vm_ray_backend._is_message_too_long(255,
                                                         output=long_output)

        http_error = "<html><h1>414 Request-URI Too Large</h1></html>"
        assert cloud_vm_ray_backend._is_message_too_long(1, output=http_error)

    def test_multiple_patterns_match_by_returncode(self):
        """Test that returncode determines which pattern to match."""
        mixed = "too long and request-uri too large"
        assert cloud_vm_ray_backend._is_message_too_long(255, output=mixed)
        assert cloud_vm_ray_backend._is_message_too_long(1, output=mixed)


class TestCloudVmRayBackendTeardownNoLock:
    """Tests for CloudVmRayBackend.teardown_no_lock() guards."""

    @staticmethod
    def _make_handle(cluster_name: str, cluster_yaml: str, has_ray: bool):

        class _FakeLaunchedResources:

            def __init__(self, cloud_obj):
                self.cloud = cloud_obj

            def assert_launchable(self):
                return self

        cloud = MagicMock()
        cloud.PROVISIONER_VERSION = (
            cloud_vm_ray_backend.clouds.ProvisionerVersion.SKYPILOT)
        launched_resources = _FakeLaunchedResources(cloud)

        handle = CloudVmRayResourceHandle(
            cluster_name=cluster_name,
            cluster_name_on_cloud=f'{cluster_name}-on-cloud',
            cluster_yaml=cluster_yaml,
            launched_nodes=1,
            launched_resources=launched_resources,
        )
        handle.provision_runtime_metadata = (
            cloud_vm_ray_backend.provision_common.ProvisionRuntimeMetadata(
                has_ray=has_ray))
        handle.close_skylet_ssh_tunnel = MagicMock()
        return handle

    def test_uses_refreshed_handle_to_avoid_stale_metadata(self):
        backend = cloud_vm_ray_backend.CloudVmRayBackend()
        stale_handle = self._make_handle('test-cluster',
                                         '/tmp/stale.yaml',
                                         has_ray=True)
        refreshed_handle = self._make_handle('test-cluster',
                                             '/tmp/refreshed.yaml',
                                             has_ray=False)

        with patch(
                'sky.backends.cloud_vm_ray_backend.requests_lib.'
                'kill_cluster_requests'), patch(
                    'sky.backends.cloud_vm_ray_backend.backend_utils.'
                    'refresh_cluster_status_handle',
                    return_value=(
                        status_lib.ClusterStatus.UP, refreshed_handle)), patch(
                            'sky.backends.cloud_vm_ray_backend.'
                            'global_user_state.'
                            'get_cluster_yaml_dict',
                            return_value={'provider': {
                            }}) as (mock_get_yaml), patch(
                                'sky.backends.cloud_vm_ray_backend'
                                '.provisioner.teardown_cluster'), patch.object(
                                    backend,
                                    'post_teardown_cleanup'), patch.object(
                                        backend,
                                        'run_on_head') as mock_run_on_head:
            backend.teardown_no_lock(stale_handle,
                                     terminate=True,
                                     refresh_cluster_status=True)

        mock_run_on_head.assert_not_called()
        mock_get_yaml.assert_called_once_with(refreshed_handle.cluster_yaml)

    @pytest.mark.parametrize('failure_point', ['kill', 'ack'])
    def test_cancellation_failure_aborts_before_cloud_teardown(
            self, failure_point):
        backend = cloud_vm_ray_backend.CloudVmRayBackend()
        handle = self._make_handle('test-cluster',
                                   '/tmp/stale.yaml',
                                   has_ray=False)
        kill_side_effect = (RuntimeError('kill failed')
                            if failure_point == 'kill' else None)
        ack_side_effect = (RuntimeError('ack timeout')
                           if failure_point == 'ack' else None)

        with patch(
                'sky.backends.cloud_vm_ray_backend.requests_lib.'
                'kill_cluster_requests',
                return_value=['launch-request'],
                side_effect=kill_side_effect), patch(
                    'sky.backends.cloud_vm_ray_backend.requests_lib.'
                    'wait_for_request_cancellation',
                    side_effect=ack_side_effect) as mock_wait, patch(
                        'sky.backends.cloud_vm_ray_backend.backend_utils.'
                        'refresh_cluster_status_handle') as mock_refresh:
            with pytest.raises(RuntimeError, match='kill failed|ack timeout'):
                backend.teardown_no_lock(handle, terminate=True)

        mock_refresh.assert_not_called()
        if failure_point == 'kill':
            mock_wait.assert_not_called()

    @pytest.mark.parametrize('failure_point', ['kill', 'ack'])
    def test_cancellation_failure_aborts_before_force_unlock(
            self, failure_point):
        backend = cloud_vm_ray_backend.CloudVmRayBackend()
        handle = self._make_handle('test-cluster',
                                   '/tmp/stale.yaml',
                                   has_ray=False)
        mock_lock = MagicMock()
        kill_side_effect = (RuntimeError('kill failed')
                            if failure_point == 'kill' else None)
        ack_side_effect = (RuntimeError('ack timeout')
                           if failure_point == 'ack' else None)

        with patch('sky.backends.cloud_vm_ray_backend.backend_utils.'
                   'check_owner_identity'), patch(
                       'sky.backends.cloud_vm_ray_backend.locks.get_lock',
                       return_value=mock_lock), patch(
                           'sky.backends.cloud_vm_ray_backend.requests_lib.'
                           'kill_cluster_requests',
                           return_value=['launch-request'],
                           side_effect=kill_side_effect), patch(
                               'sky.backends.cloud_vm_ray_backend.requests_lib.'
                               'wait_for_request_cancellation',
                               side_effect=ack_side_effect) as mock_wait, \
             patch.object(
                                       backend,
                                       'teardown_no_lock') as mock_teardown:
            with pytest.raises(RuntimeError, match='kill failed|ack timeout'):
                backend._teardown(handle, terminate=True)

        mock_lock.force_unlock.assert_not_called()
        mock_teardown.assert_not_called()
        if failure_point == 'kill':
            mock_wait.assert_not_called()


class TestNewHandleRuntimeMetadata:
    """Runtime metadata a freshly constructed handle starts with."""

    def test_new_handle_has_no_runtime_established(self):
        """A new handle is created before provisioning, so it claims no Ray.

        Otherwise teardown of a cluster that crashed or recovered during
        provisioning attempts ``ray stop`` on a runtime that was never set
        up.
        """
        handle = CloudVmRayResourceHandle(
            cluster_name='test-cluster',
            cluster_name_on_cloud='test-cluster-abc',
            cluster_yaml=None,
            launched_nodes=1,
            launched_resources=MagicMock(),
        )
        metadata = handle.provision_runtime_metadata
        assert (metadata.has_ray, metadata.has_skylet, metadata.has_job_queue,
                metadata.ssh_available) == (False, False, False, False)


def test_compute_tpu_mig_scale_down_rejected_before_teardown():
    launched_resources = resources.Resources(infra='gcp/us-east5/us-east5-a',
                                             instance_type='ct6e-standard-4t')
    handle = CloudVmRayResourceHandle(
        cluster_name='tpu-cluster',
        cluster_name_on_cloud='tpu-cluster-cloud',
        cluster_yaml='/tmp/tpu-cluster.yml',
        launched_nodes=4,
        launched_resources=launched_resources,
    )
    resize_task = task.Task(num_nodes=3, resources=launched_resources)
    backend = cloud_vm_ray_backend.CloudVmRayBackend()

    with patch.object(
            cloud_vm_ray_backend.global_user_state,
            'get_cluster_yaml_dict',
            return_value={'provider': {
                'use_managed_instance_group': True,
            }}), patch.object(
                cloud_vm_ray_backend.provision_lib,
                'terminate_instances') as terminate_instances, patch.object(
                    backend, 'run_on_head') as run_on_head:
        with pytest.raises(exceptions.NotSupportedError,
                           match='smaller topology is validated'):
            backend._handle_resize_pre_provision(
                handle,
                resize_task,
                'tpu-cluster',
                cluster_status=status_lib.ClusterStatus.UP)

    terminate_instances.assert_not_called()
    run_on_head.assert_not_called()


def test_slurm_execute_retries_transient_submission_connection_failure():
    backend = cloud_vm_ray_backend.CloudVmRayBackend()
    handle = MagicMock()
    handle.provision_runtime_metadata.run_started = False
    handle.launched_resources.cloud = clouds.Slurm()
    handle.num_ips_per_node = 1
    test_task = task.Task(run='true')
    valid_resource = resources.Resources()
    connection_error = exceptions.CommandError(255, 'ssh isambard',
                                               'Failed to submit job.',
                                               'Connection reset by peer')
    home_error = ValueError(
        'Failed to get remote home directory: Connection reset by peer')
    submission_ids = []

    def add_job_side_effect(*args, **kwargs):
        del args, kwargs
        submission_ids.append(backend._submission_id)
        return 1, '/tmp/logs'

    with patch.object(
            backend, 'check_resources_fit_cluster',
            return_value=valid_resource), patch.object(
                backend, '_add_job',
                side_effect=add_job_side_effect) as add_job, patch.object(
                    backend,
                    '_execute_task_one_node',
                    side_effect=[home_error, connection_error,
                                 None]) as execute_task, patch.object(
                                     cloud_vm_ray_backend.time,
                                     'sleep') as sleep:
        assert backend._execute(handle, test_task) == 1

    assert add_job.call_count == 3
    assert execute_task.call_count == 3
    assert sleep.call_args_list == [call(60), call(60)]
    assert len(set(submission_ids)) == 1


def test_reused_backend_gets_fresh_local_submission_id():
    backend = cloud_vm_ray_backend.CloudVmRayBackend()
    handle = MagicMock()
    handle.provision_runtime_metadata.run_started = False
    handle.launched_resources.cloud = clouds.Slurm()
    handle.num_ips_per_node = 1
    test_task = task.Task(run='true')
    valid_resource = resources.Resources()
    submission_ids = []

    def add_job_side_effect(*args, **kwargs):
        del args, kwargs
        submission_ids.append(backend._submission_id)
        return len(submission_ids), '/tmp/logs'

    with patch.object(cloud_vm_ray_backend.common_utils,
                      'get_current_request_id',
                      return_value='dummy-request-id'), patch.object(
                          backend,
                          'check_resources_fit_cluster',
                          return_value=valid_resource), patch.object(
                              backend,
                              '_add_job',
                              side_effect=add_job_side_effect), patch.object(
                                  backend, '_execute_task_one_node'):
        assert backend._execute(handle, test_task) == 1
        assert backend._execute(handle, test_task) == 2

    assert len(submission_ids) == 2
    assert submission_ids[0] != submission_ids[1]


def test_slurm_add_job_always_uses_submission_id():
    backend = cloud_vm_ray_backend.CloudVmRayBackend()
    handle = MagicMock()
    handle.is_grpc_enabled_with_flag = True
    handle.launched_resources.cloud = clouds.Slurm()

    with patch.object(
            cloud_vm_ray_backend.job_lib.JobLibCodeGen,
            'add_job',
            return_value='add-job-code') as add_job_codegen, patch.object(
                backend,
                'run_on_head',
                return_value=(0, 'Job ID: 1\nLog Dir: /tmp/logs\n', '')):
        assert backend._add_job(handle, 'train', '4xGH200',
                                '{}') == (1, '/tmp/logs')

    assert add_job_codegen.call_args.kwargs[
        'submission_id'] == backend.run_timestamp


def test_slurm_queue_is_synchronous_and_enqueue_only():
    backend = cloud_vm_ray_backend.CloudVmRayBackend()
    handle = MagicMock()
    handle.is_grpc_enabled_with_flag = True
    handle.launched_resources.cloud = clouds.Slurm()
    handle.cluster_name = 'isambard'

    with patch.object(
            cloud_vm_ray_backend.job_lib.JobLibCodeGen,
            'queue_job',
            return_value='queue-code') as queue_job_codegen, patch.object(
                cloud_vm_ray_backend.job_lib.JobLibCodeGen,
                'wait_for_job') as wait_for_job, patch.object(
                    cloud_vm_ray_backend.backend_utils,
                    'is_command_length_over_limit',
                    return_value=False), patch.object(
                        cloud_vm_ray_backend.controller_utils.Controllers,
                        'from_name',
                        return_value=None), patch.object(
                            backend, 'run_on_head',
                            return_value=(0, '', '')) as run_on_head:
        backend._exec_code_on_head(handle, 'print("train")', 1)

    queue_job_codegen.assert_called_once()
    assert queue_job_codegen.call_args.kwargs['schedule'] is False
    queued_cmd = queue_job_codegen.call_args.args[1]
    marker = cloud_vm_ray_backend.job_lib.JOB_CMD_IDENTIFIER.format(1)
    assert queued_cmd.count(marker) == 2
    wait_for_job.assert_not_called()
    assert run_on_head.call_args.kwargs['run_in_background'] is False


class TestProvisionClusterLockParking:
    """_provision on lock contention: park as WAITING only in request ctx."""

    def _run_provision(self,
                       monkeypatch,
                       in_request_context,
                       locked_provision_mock,
                       is_launched_by_jobs_controller=False):
        backend = cloud_vm_ray_backend.CloudVmRayBackend()
        backend._is_launched_by_jobs_controller = (
            is_launched_by_jobs_controller)
        monkeypatch.setattr('sky.backends.backend_utils.check_rsync_installed',
                            lambda: None)
        monkeypatch.setattr('sky.backends.backend_utils.check_owner_identity',
                            lambda cluster_name: None)
        monkeypatch.setattr('sky.utils.common_utils.is_in_request_context',
                            lambda: in_request_context)
        monkeypatch.setattr('sky.utils.rich_utils.force_update_status',
                            lambda msg: None)
        monkeypatch.setattr(backend, '_locked_provision', locked_provision_mock)
        return backend._provision(MagicMock(),
                                  None,
                                  dryrun=False,
                                  stream_logs=False,
                                  cluster_name='test-cluster')

    def test_parks_on_lock_contention_in_request_context(self, monkeypatch):
        locked_provision = MagicMock(side_effect=locks.LockTimeout('locked'))
        with pytest.raises(exceptions.ExecutionPausedError) as exc_info:
            self._run_provision(monkeypatch,
                                in_request_context=True,
                                locked_provision_mock=locked_provision)
        assert 'test-cluster' in str(exc_info.value)
        assert (exc_info.value.retry_wait_seconds ==
                cloud_vm_ray_backend._CLUSTER_LOCK_RETRY_GAP_SECONDS)
        condition = exc_info.value.continue_condition
        assert isinstance(condition, locks.LockAcquirableCondition)
        assert condition._lock_id == 'test-cluster_status'
        assert locked_provision.call_count == 1

    def test_blocks_on_lock_contention_outside_request_context(
            self, monkeypatch):
        sentinel = (MagicMock(), False)
        locked_provision = MagicMock(
            side_effect=[locks.LockTimeout('locked'), sentinel])
        result = self._run_provision(monkeypatch,
                                     in_request_context=False,
                                     locked_provision_mock=locked_provision)
        assert result is sentinel
        assert locked_provision.call_count == 2

    def test_parks_for_jobs_controller_launch_in_request_context(
            self, monkeypatch):
        # A jobs-controller launch running as a scheduler-managed request
        # (consolidation mode submits launches via the SDK like any other
        # request) must park like any other request: blocking instead would
        # pin one executor worker per lock contender (e.g. duplicate launch
        # requests for the same cluster after controller retries), starving
        # the worker pool at scale.
        locked_provision = MagicMock(side_effect=locks.LockTimeout('locked'))
        with pytest.raises(exceptions.ExecutionPausedError) as exc_info:
            self._run_provision(monkeypatch,
                                in_request_context=True,
                                locked_provision_mock=locked_provision,
                                is_launched_by_jobs_controller=True)
        condition = exc_info.value.continue_condition
        assert isinstance(condition, locks.LockAcquirableCondition)
        assert locked_provision.call_count == 1

    def test_blocks_for_jobs_controller_launch_outside_request_context(
            self, monkeypatch):
        # Without a request context there is no scheduler to park/resume the
        # request, so the blocking behavior is kept regardless of the
        # jobs-controller flag.
        sentinel = (MagicMock(), False)
        locked_provision = MagicMock(
            side_effect=[locks.LockTimeout('locked'), sentinel])
        result = self._run_provision(monkeypatch,
                                     in_request_context=False,
                                     locked_provision_mock=locked_provision,
                                     is_launched_by_jobs_controller=True)
        assert result is sentinel
        assert locked_provision.call_count == 2


class TestSlurmContainerImageBackfill:
    """Backfilling provider.container_image for pre-upgrade Slurm clusters."""

    @staticmethod
    def _handle(cloud, image):
        launched = MagicMock()
        launched.cloud = MagicMock(spec=cloud)
        launched.extract_docker_image.return_value = image
        return CloudVmRayResourceHandle(
            cluster_name='test-cluster',
            cluster_name_on_cloud='test-cluster-abc',
            cluster_yaml='test-cluster.yml',
            launched_nodes=1,
            launched_resources=launched,
        )

    def test_backfills_legacy_container_cluster(self):
        handle = self._handle(clouds.Slurm, 'ubuntu:24.04')
        with patch('sky.global_user_state.get_cluster_yaml_dict',
                   return_value={'provider': {'cluster': 'c'}}), \
             patch('sky.global_user_state.set_cluster_yaml') as mock_set:
            handle._maybe_backfill_slurm_container_image()
        mock_set.assert_called_once()
        name, written = mock_set.call_args.args
        assert name == 'test-cluster'
        from sky.utils import yaml_utils
        assert (yaml_utils.safe_load(written)['provider']['container_image'] ==
                'ubuntu:24.04')

    def test_noop_when_already_present(self):
        handle = self._handle(clouds.Slurm, 'ubuntu:24.04')
        with patch('sky.global_user_state.get_cluster_yaml_dict',
                   return_value={'provider': {
                       'container_image': 'ubuntu:24.04'
                   }}), \
             patch('sky.global_user_state.set_cluster_yaml') as mock_set:
            handle._maybe_backfill_slurm_container_image()
        mock_set.assert_not_called()

    def test_reconciles_stale_value(self):
        handle = self._handle(clouds.Slurm, 'ubuntu:24.04')
        with patch('sky.global_user_state.get_cluster_yaml_dict',
                   return_value={'provider': {
                       'container_image': 'old:1'
                   }}), \
             patch('sky.global_user_state.set_cluster_yaml') as mock_set:
            handle._maybe_backfill_slurm_container_image()
        _, written = mock_set.call_args.args
        from sky.utils import yaml_utils
        assert (yaml_utils.safe_load(written)['provider']['container_image'] ==
                'ubuntu:24.04')

    def test_noop_for_non_container_slurm(self):
        handle = self._handle(clouds.Slurm, None)
        with patch('sky.global_user_state.get_cluster_yaml_dict') as mock_get, \
             patch('sky.global_user_state.set_cluster_yaml') as mock_set:
            handle._maybe_backfill_slurm_container_image()
        mock_get.assert_not_called()
        mock_set.assert_not_called()

    def test_noop_for_non_slurm_cloud(self):
        handle = self._handle(clouds.AWS, 'ubuntu:24.04')
        with patch('sky.global_user_state.get_cluster_yaml_dict') as mock_get, \
             patch('sky.global_user_state.set_cluster_yaml') as mock_set:
            handle._maybe_backfill_slurm_container_image()
        mock_get.assert_not_called()
        mock_set.assert_not_called()


def _head(skylet_port=None):
    return provision_common.InstanceInfo(instance_id='h',
                                         internal_ip='10.0.0.1',
                                         external_ip=None,
                                         tags={},
                                         skylet_port=skylet_port)


class TestSkyletPort:
    """The skylet tunnel dials the head's skylet, not a constant.

    Two hostNetwork heads on one node used to share the default: the second
    skylet listened on the next free port while the server dialed the
    default, so one cluster's jobs ran on the other.
    """

    def _handle(self, head, with_info=True):
        handle = CloudVmRayResourceHandle(
            **TestCloudVmRayBackendGetGrpcChannel.MOCK_HANDLE_KWARGS)
        if with_info:
            info = MagicMock()
            info.get_head_instance.return_value = head
            handle.cached_cluster_info = info
        else:
            handle.cached_cluster_info = None
        return handle

    def test_an_assigned_port_is_dialed(self):
        assert self._handle(_head(29070)).skylet_port == 29070

    def test_no_assignment_is_the_default(self):
        assert (self._handle(
            _head()).skylet_port == skylet_constants.SKYLET_GRPC_PORT)

    def test_a_head_pickled_before_the_field_is_the_default(self):
        old = types.SimpleNamespace(instance_id='h', ssh_port=22)
        assert (
            self._handle(old).skylet_port == skylet_constants.SKYLET_GRPC_PORT)

    def test_no_cluster_info_is_the_default(self):
        assert (self._handle(
            None,
            with_info=False).skylet_port == skylet_constants.SKYLET_GRPC_PORT)

    def test_the_tunnel_dials_the_heads_port(self, monkeypatch):
        handle = self._handle(_head(29070))
        monkeypatch.setattr(handle, 'get_command_runners',
                            lambda: [MagicMock()])
        dialed = []

        class _Stop(Exception):
            pass

        def fake_tunnel(runner, port_pair):
            del runner
            dialed.append(port_pair)
            raise _Stop()

        monkeypatch.setattr(backend_utils, 'open_ssh_tunnel', fake_tunnel)
        with pytest.raises(_Stop):
            handle._open_and_update_skylet_tunnel()  # pylint: disable=protected-access
        assert dialed[0][1] == 29070


class TestProvisionExistingClusterError:
    """A provisioning error on an existing cluster reaches the user."""

    @staticmethod
    def _provision(prev_status, retry_zones_error=None, features_error=None):
        provisioner = object.__new__(cloud_vm_ray_backend.RetryingVmProvisioner)
        provisioner._dag = MagicMock()  # pylint: disable=protected-access
        provisioner._optimize_target = MagicMock()  # pylint: disable=protected-access
        provisioner._requested_features = set()  # pylint: disable=protected-access
        provisioner._blocked_resources = set()  # pylint: disable=protected-access
        provisioner._is_launched_by_jobs_controller = False  # pylint: disable=protected-access
        provisioner.log_dir = '/tmp'
        provisioner._retry_zones = MagicMock(side_effect=retry_zones_error)  # pylint: disable=protected-access
        to_provision = MagicMock()
        setattr(to_provision, 'assert_launchable', lambda: to_provision)
        to_provision.cloud.check_features_are_supported.side_effect = (
            features_error)
        config = cloud_vm_ray_backend.RetryingVmProvisioner.ToProvisionConfig(
            'c1',
            to_provision,
            1,
            prev_status,
            MagicMock(),
            prev_cluster_ever_up=prev_status is not None,
            prev_config_hash=None)
        task_mock = MagicMock()
        task_mock.is_controller_task.return_value = False
        return provisioner.provision_with_retries(
            task_mock,
            config,
            dryrun=False,
            stream_logs=False,
            skip_unnecessary_provisioning=False)

    _ERRORS = [
        pytest.param(
            dict(retry_zones_error=exceptions.ResourcesUnavailableError(
                'invalid cloud credentials: expired')),
            'expired',
            id='retry_zones'),
        pytest.param(dict(
            features_error=exceptions.NotSupportedError('no autodown here')),
                     'no autodown here',
                     id='features'),
    ]

    @pytest.mark.parametrize('error,message', _ERRORS)
    @pytest.mark.parametrize(
        'status',
        [status_lib.ClusterStatus.UP, status_lib.ClusterStatus.STOPPED])
    def test_up_or_stopped_cluster_reports_the_error(self, status, error,
                                                     message):
        with pytest.raises(exceptions.ResourcesUnavailableError) as e:
            self._provision(status, **error)
        assert e.value.no_failover
        assert message in str(e.value)

    @pytest.mark.parametrize('error,message', _ERRORS)
    def test_init_cluster_still_fails_over(self, error, message):
        del message
        with patch.object(cloud_vm_ray_backend.optimizer.Optimizer,
                          'optimize',
                          side_effect=exceptions.ResourcesUnavailableError(
                              'nothing left')) as optimize:
            with pytest.raises(exceptions.ResourcesUnavailableError) as e:
                self._provision(status_lib.ClusterStatus.INIT, **error)
        optimize.assert_called_once()
        assert not e.value.no_failover
