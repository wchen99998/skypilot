"""Tests for retry-safe Skylet job submission."""

import os
import shlex
import sqlite3
from unittest import mock

import pytest

from sky.skylet import events
from sky.skylet import job_lib
from sky.utils.db import db_utils


@pytest.fixture
def job_db(tmp_path, monkeypatch):
    db = db_utils.SQLiteConn(str(tmp_path / 'jobs.db'), job_lib.create_table)
    monkeypatch.setattr(job_lib, '_DB', db)
    monkeypatch.setattr(job_lib, '_JOB_STATUS_LOCK',
                        str(tmp_path / '.job_{}.lock'))
    yield db
    db.conn.close()


def test_add_job_replay_returns_same_job(job_db):
    first = job_lib.add_job('train',
                            'user',
                            'worker-timestamp',
                            '4xGH200',
                            submission_id='submission-token')
    second = job_lib.add_job('train',
                             'user',
                             'worker-timestamp',
                             '4xGH200',
                             submission_id='submission-token')

    assert second == first
    count = job_db.cursor.execute(
        'SELECT COUNT(*) FROM jobs WHERE run_timestamp=(?)',
        ('worker-timestamp',)).fetchone()[0]
    assert count == 1


def test_legacy_add_job_reused_timestamp_creates_new_job(job_db):
    first, _ = job_lib.add_job('first', 'user', 'worker-timestamp', '1xGPU')
    second, _ = job_lib.add_job('second', 'user', 'worker-timestamp', '1xGPU')

    assert second != first
    assert job_db.cursor.execute(
        'SELECT COUNT(*) FROM jobs WHERE run_timestamp=(?)',
        ('worker-timestamp',)).fetchone()[0] == 2


def test_submission_id_survives_new_run_timestamp(job_db):
    first = job_lib.add_job('train',
                            'user',
                            'first-worker-timestamp',
                            '4xGH200',
                            submission_id='api-request-id')
    second = job_lib.add_job('train',
                             'user',
                             'replacement-worker-timestamp',
                             '4xGH200',
                             submission_id='api-request-id')

    assert second == first
    count = job_db.cursor.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]
    assert count == 1


def test_distinct_submission_ids_do_not_share_run_timestamp(job_db):
    first, _ = job_lib.add_job('first',
                               'user',
                               'backend-timestamp',
                               '4xGH200',
                               submission_id='first-request')
    second, _ = job_lib.add_job('second',
                                'user',
                                'backend-timestamp',
                                '4xGH200',
                                submission_id='second-request')

    assert second != first
    count = job_db.cursor.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]
    assert count == 2


def test_incomplete_legacy_migration_is_recovered(tmp_path):
    db_path = tmp_path / 'legacy-jobs.db'
    missing_pid = 2**31 - 1
    conn = sqlite3.connect(db_path)
    conn.execute(
        'CREATE TABLE jobs(job_id INTEGER PRIMARY KEY, status TEXT, pid INTEGER)'
    )
    conn.execute('INSERT INTO jobs VALUES (1, ?, ?)',
                 (job_lib.JobStatus.RUNNING.value, missing_pid))
    conn.execute(
        'CREATE TABLE legacy_scheduler_jobs(job_id INTEGER PRIMARY KEY)')
    conn.commit()
    conn.close()

    migrated_db = db_utils.SQLiteConn(str(db_path), job_lib.create_table)

    assert migrated_db.cursor.execute(
        'SELECT job_id FROM legacy_scheduler_jobs').fetchall() == [(1,)]
    assert migrated_db.cursor.execute(
        'SELECT pid, pid_create_time FROM legacy_scheduler_jobs').fetchone(
        ) == (missing_pid, None)
    assert migrated_db.cursor.execute(
        'SELECT name FROM schema_migrations WHERE name=(?)',
        (job_lib._SLURM_SCHEDULER_MIGRATION,)).fetchone() is not None
    migrated_db.conn.close()


def test_migration_ignores_terminal_reused_pid(tmp_path):
    db_path = tmp_path / 'terminal-legacy-jobs.db'
    conn = sqlite3.connect(db_path)
    conn.execute(
        'CREATE TABLE jobs(job_id INTEGER PRIMARY KEY, status TEXT, pid INTEGER)'
    )
    conn.execute('INSERT INTO jobs VALUES (1, ?, ?)',
                 (job_lib.JobStatus.SUCCEEDED.value, os.getpid()))
    conn.commit()
    conn.close()

    migrated_db = db_utils.SQLiteConn(str(db_path), job_lib.create_table)

    assert migrated_db.cursor.execute(
        'SELECT pid, pid_create_time FROM legacy_scheduler_jobs '
        'WHERE job_id=1').fetchone() is None
    migrated_db.conn.close()


def test_migration_captures_terminal_matching_legacy_driver(tmp_path):
    db_path = tmp_path / 'terminal-live-legacy-jobs.db'
    conn = sqlite3.connect(db_path)
    conn.execute(
        'CREATE TABLE jobs(job_id INTEGER PRIMARY KEY, status TEXT, pid INTEGER)'
    )
    conn.execute('INSERT INTO jobs VALUES (1, ?, 123)',
                 (job_lib.JobStatus.SUCCEEDED.value,))
    conn.commit()
    conn.close()

    with mock.patch(
            'sky.skylet.job_lib._get_process_create_time',
            return_value=10.0), mock.patch(
                'sky.skylet.job_lib._is_legacy_job_driver_process_running',
                return_value=True):
        migrated_db = db_utils.SQLiteConn(str(db_path), job_lib.create_table)

    assert migrated_db.cursor.execute(
        'SELECT pid, pid_create_time FROM legacy_scheduler_jobs '
        'WHERE job_id=1').fetchone() == (123, 10.0)
    migrated_db.conn.close()


def test_new_slurm_submission_waits_for_legacy_driver(job_db):
    legacy_id, _ = job_lib.add_job('legacy', 'user', 'legacy-run', '4xGH200')
    job_db.cursor.execute('UPDATE jobs SET status=(?) WHERE job_id=(?)',
                          (job_lib.JobStatus.RUNNING.value, legacy_id))
    job_db.cursor.execute(
        'INSERT INTO legacy_scheduler_jobs(job_id) VALUES (?)', (legacy_id,))
    job_db.conn.commit()

    with pytest.raises(RuntimeError, match='pre-v43 Slurm runtime'):
        job_lib.add_job('new',
                        'user',
                        'new-run',
                        '4xGH200',
                        submission_id='new-request')

    job_lib.set_status(legacy_id, job_lib.JobStatus.SUCCEEDED)
    new_id, _ = job_lib.add_job('new',
                                'user',
                                'new-run',
                                '4xGH200',
                                submission_id='new-request')
    assert new_id != legacy_id


def test_new_slurm_submission_waits_for_terminal_live_legacy_driver(job_db):
    legacy_id, _ = job_lib.add_job('legacy', 'user', 'legacy-run', '4xGH200')
    job_db.cursor.execute(
        'UPDATE jobs SET status=(?), pid=123 WHERE job_id=(?)',
        (job_lib.JobStatus.SUCCEEDED.value, legacy_id))
    job_db.cursor.execute('INSERT INTO legacy_scheduler_jobs VALUES (?, ?, ?)',
                          (legacy_id, 123, 1.0))
    job_db.conn.commit()

    with mock.patch('sky.skylet.job_lib._is_legacy_job_driver_process_running',
                    return_value=True), pytest.raises(
                        RuntimeError, match='pre-v43 Slurm runtime'):
        job_lib.add_job('new',
                        'user',
                        'new-run',
                        '4xGH200',
                        submission_id='new-request')

    with mock.patch('sky.skylet.job_lib._is_legacy_job_driver_process_running',
                    return_value=False):
        new_id, _ = job_lib.add_job('new',
                                    'user',
                                    'new-run',
                                    '4xGH200',
                                    submission_id='new-request')
    assert new_id != legacy_id


def test_legacy_gate_tracks_driver_launched_after_migration(job_db):
    legacy_id, _ = job_lib.add_job('legacy', 'user', 'legacy-run', '4xGH200')
    job_db.cursor.execute(
        'UPDATE jobs SET status=(?), pid=123 WHERE job_id=(?)',
        (job_lib.JobStatus.SUCCEEDED.value, legacy_id))
    job_db.cursor.execute('INSERT INTO legacy_scheduler_jobs VALUES (?, ?, ?)',
                          (legacy_id, 0, None))
    job_db.conn.commit()

    with mock.patch(
            'sky.skylet.job_lib._get_process_create_time',
            return_value=10.0), mock.patch(
                'sky.skylet.job_lib._is_legacy_job_driver_process_running',
                return_value=True), pytest.raises(
                    RuntimeError, match='pre-v43 Slurm runtime'):
        job_lib.add_job('new',
                        'user',
                        'new-run',
                        '4xGH200',
                        submission_id='new-request')


def test_legacy_driver_identity_uses_pid_create_time():
    process = mock.MagicMock()
    process.is_running.return_value = True
    process.create_time.return_value = 10.0
    process.cmdline.return_value = ['python', '/tmp/sky_job_10']

    with mock.patch.object(job_lib.psutil, 'Process', return_value=process):
        assert job_lib._is_legacy_job_driver_process_running(123, 10, 10.0)
        assert not job_lib._is_legacy_job_driver_process_running(123, 10, 11.0)
        assert not job_lib._is_legacy_job_driver_process_running(123, 1, 10.0)


def test_add_job_codegen_carries_submission_id():
    code = job_lib.JobLibCodeGen.add_job('train',
                                         'user',
                                         'run-timestamp',
                                         '4xGH200',
                                         '{}',
                                         submission_id='api-request-id')

    python_code = shlex.split(code)[-1]
    compile(python_code, '<add-job>', 'exec')
    assert "submission_id='api-request-id'" in python_code
    assert 'SKYLET_VERSION) < 44' in python_code


def test_add_job_codegen_preserves_legacy_signature():
    code = job_lib.JobLibCodeGen.add_job('train', 'user', 'run-timestamp',
                                         '1xGPU', '{}')
    python_code = shlex.split(code)[-1]

    assert 'submission_id=' not in python_code
    compile(python_code, '<legacy-add-job>', 'exec')


def test_add_job_codegen_rejects_old_slurm_runtime():
    code = job_lib.JobLibCodeGen.add_job('train',
                                         'user',
                                         'run-timestamp',
                                         '4xGH200',
                                         '{}',
                                         submission_id='api-request-id')
    python_code = shlex.split(code)[-1]

    with mock.patch.object(job_lib.constants, 'SKYLET_VERSION',
                           '42'), mock.patch.object(job_lib,
                                                    'add_job') as add_job:
        with pytest.raises(RuntimeError, match='SkyPilot runtime is too old'):
            exec(python_code)  # pylint: disable=exec-used

    add_job.assert_not_called()


def test_queue_codegen_preserves_legacy_call_and_gates_slurm():
    legacy_code = shlex.split(
        job_lib.JobLibCodeGen.queue_job(1, 'train-command'))[-1]
    slurm_code = shlex.split(
        job_lib.JobLibCodeGen.queue_job(1, 'train-command', schedule=False))[-1]

    assert 'schedule=' not in legacy_code
    assert 'SKYLET_VERSION) < 44' in slurm_code
    assert 'schedule=False' in slurm_code


def test_queue_replay_does_not_launch_job_twice(job_db):
    job_id, _ = job_lib.add_job('train', 'user', 'submission-token', '4xGH200')

    with mock.patch(
            'sky.skylet.job_lib.subprocess_utils.launch_new_process_tree',
            return_value=123) as launch, mock.patch(
                'sky.skylet.job_lib._is_job_driver_process_running',
                return_value=True):
        job_lib.scheduler.queue(job_id, 'train-command')
        job_lib.scheduler.queue(job_id, 'train-command')
        job_lib.set_job_started(job_id)
        job_lib.scheduler.schedule_step()
        job_lib.scheduler.queue(job_id, 'train-command')

    launch.assert_called_once()
    assert 'claim_job' in launch.call_args.args[0]
    assert 'train-command' in launch.call_args.args[0]
    assert job_lib.get_status(job_id) == job_lib.JobStatus.RUNNING
    pending_count = job_db.cursor.execute(
        'SELECT COUNT(*) FROM pending_jobs WHERE job_id=(?)',
        (job_id,)).fetchone()[0]
    assert pending_count == 0


def test_slurm_queue_is_launched_only_by_skylet(job_db):
    job_id, _ = job_lib.add_job('train', 'user', 'submission-token', '4xGH200')

    with mock.patch(
            'sky.skylet.job_lib.subprocess_utils.launch_new_process_tree',
            return_value=123) as launch, mock.patch(
                'sky.skylet.job_lib._is_job_driver_process_running',
                return_value=True):
        job_lib.scheduler.queue(job_id, 'train-command', schedule=False)
        job_lib.scheduler.queue(job_id, 'train-command', schedule=False)
        launch.assert_not_called()

        job_lib.scheduler.schedule_step()
        job_lib.scheduler.schedule_step()

    launch.assert_called_once()
    assert 'claim_job' in launch.call_args.args[0]
    assert 'train-command' in launch.call_args.args[0]
    assert job_lib.get_status(job_id) == job_lib.JobStatus.PENDING


def test_job_driver_claim_is_atomic(job_db):
    job_id, _ = job_lib.add_job('train', 'user', 'submission-token', '4xGH200')
    job_lib.scheduler.queue(job_id, 'train-command', schedule=False)

    job_db.cursor.execute('UPDATE pending_jobs SET submit=-1 WHERE job_id=(?)',
                          (job_id,))
    job_db.conn.commit()
    assert job_lib.claim_job(job_id, 111)
    assert not job_lib.claim_job(job_id, 222)
    pid = job_db.cursor.execute('SELECT pid FROM jobs WHERE job_id=(?)',
                                (job_id,)).fetchone()[0]
    submit = job_db.cursor.execute(
        'SELECT submit FROM pending_jobs WHERE job_id=(?)',
        (job_id,)).fetchone()[0]
    assert pid == 111
    assert submit == -2

    job_lib.mark_job_driver_started(job_id)
    submit = job_db.cursor.execute(
        'SELECT submit FROM pending_jobs WHERE job_id=(?)',
        (job_id,)).fetchone()[0]
    assert submit > 0


def test_dead_claimed_driver_is_relaunched(job_db):
    job_id, _ = job_lib.add_job('train', 'user', 'submission-token', '4xGH200')
    job_lib.scheduler.queue(job_id, 'train-command', schedule=False)

    with mock.patch(
            'sky.skylet.job_lib.subprocess_utils.launch_new_process_tree',
            side_effect=[123, 456]) as launch, mock.patch(
                'sky.skylet.job_lib._is_job_driver_process_running',
                return_value=False):
        job_lib.scheduler.schedule_step()
        assert job_lib.claim_job(job_id, 123)
        job_lib.scheduler.schedule_step()

    assert launch.call_count == 2
    pid = job_db.cursor.execute('SELECT pid FROM jobs WHERE job_id=(?)',
                                (job_id,)).fetchone()[0]
    assert pid == 456


def test_only_allocation_skylet_schedules_on_slurm(job_db, monkeypatch):
    job_id, _ = job_lib.add_job('train', 'user', 'submission-token', '4xGH200')
    job_lib.scheduler.queue(job_id, 'train-command', schedule=False)
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.delenv(job_lib.constants.SLURM_SKYLET_OWNER_ENV_VAR,
                       raising=False)

    with mock.patch(
            'sky.skylet.job_lib.subprocess_utils.launch_new_process_tree',
            return_value=123,
    ) as launch:
        job_lib.scheduler.schedule_step()
        launch.assert_not_called()

        monkeypatch.setenv(job_lib.constants.SLURM_SKYLET_OWNER_ENV_VAR, '1')
        job_lib.scheduler.schedule_step()

    launch.assert_called_once()


def test_slurm_cancel_does_not_launch_next_job(job_db):
    first_id, _ = job_lib.add_job('first', 'user', 'first-token', '4xGH200')
    second_id, _ = job_lib.add_job('second', 'user', 'second-token', '4xGH200')
    job_lib.scheduler.queue(first_id, 'first-command', schedule=False)
    job_lib.scheduler.queue(second_id, 'second-command', schedule=False)

    with mock.patch(
            'sky.skylet.job_lib.subprocess_utils.launch_new_process_tree'
    ) as launch:
        cancelled = job_lib.cancel_jobs([first_id], schedule=False)

    assert cancelled == [first_id]
    launch.assert_not_called()
    assert job_lib.get_status(first_id) == job_lib.JobStatus.CANCELLED
    assert job_lib.get_status(second_id) == job_lib.JobStatus.PENDING


def test_cancel_before_queue_prevents_replayed_queue_launch(job_db):
    job_id, _ = job_lib.add_job('train', 'user', 'submission-token', '4xGH200')

    cancelled = job_lib.cancel_jobs([job_id], schedule=False)
    with mock.patch(
            'sky.skylet.job_lib.subprocess_utils.launch_new_process_tree'
    ) as launch:
        job_lib.scheduler.queue(job_id, 'train-command', schedule=False)
        job_lib.scheduler.schedule_step()

    assert cancelled == [job_id]
    assert job_lib.get_status(job_id) == job_lib.JobStatus.CANCELLED
    launch.assert_not_called()
    assert job_db.cursor.execute('SELECT 1 FROM pending_jobs WHERE job_id=(?)',
                                 (job_id,)).fetchone() is None


def test_slurm_skylet_schedules_immediately_on_start(monkeypatch):
    monkeypatch.setattr(events, '_IS_SLURM_ALLOCATION', True)
    event = events.JobSchedulerEvent()

    with mock.patch.object(job_lib.scheduler, 'schedule_step') as schedule_step:
        event.start()

    schedule_step.assert_called_once_with(force_update_jobs=True)


def test_add_job_replay_revives_unqueued_reservation(job_db):
    job_id, _ = job_lib.add_job('train',
                                'user',
                                'worker-timestamp',
                                '4xGH200',
                                submission_id='submission-token')
    job_lib.set_status(job_id, job_lib.JobStatus.FAILED_DRIVER)

    replayed_job_id, _ = job_lib.add_job('train',
                                         'user',
                                         'worker-timestamp',
                                         '4xGH200',
                                         submission_id='submission-token')
    assert job_lib.get_status(job_id) == job_lib.JobStatus.FAILED_DRIVER

    with mock.patch(
            'sky.skylet.job_lib.subprocess_utils.launch_new_process_tree',
            return_value=123) as launch:
        job_lib.scheduler.queue(replayed_job_id, 'train-command')

    assert replayed_job_id == job_id
    launch.assert_called_once()
    assert 'claim_job' in launch.call_args.args[0]
    assert 'train-command' in launch.call_args.args[0]
    assert job_lib.get_status(job_id) == job_lib.JobStatus.PENDING
    count = job_db.cursor.execute(
        'SELECT COUNT(*) FROM jobs WHERE run_timestamp=(?)',
        ('worker-timestamp',)).fetchone()[0]
    assert count == 1
