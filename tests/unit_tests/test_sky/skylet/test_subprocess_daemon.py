"""Unit tests for the subprocess cleanup daemon."""

from unittest import mock

import psutil

from sky.skylet import subprocess_daemon


def test_is_process_running():
    process = mock.Mock()
    process.is_running.return_value = True
    process.status.return_value = psutil.STATUS_RUNNING

    assert subprocess_daemon.is_process_running(process)


def test_zombie_process_is_not_running():
    process = mock.Mock()
    process.is_running.return_value = True
    process.status.return_value = psutil.STATUS_ZOMBIE

    assert not subprocess_daemon.is_process_running(process)


def test_missing_process_is_not_running():
    process = mock.Mock()
    process.is_running.side_effect = psutil.NoSuchProcess(123)

    assert not subprocess_daemon.is_process_running(process)
