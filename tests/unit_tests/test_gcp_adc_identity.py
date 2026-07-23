import configparser
import os
import sqlite3
from unittest import mock

import pytest

from sky import exceptions
from sky.clouds import gcp as gcp_cloud
from sky.clouds.gcp import GCP
from sky.clouds.gcp import GCPIdentityType


def _credential_class(module, name='Credentials', bases=(object,)):
    return type(name, bases, {'__module__': module})


@pytest.mark.parametrize(
    ('credential', 'expected'),
    [
        (_credential_class('google.oauth2.credentials')(),
         GCPIdentityType.AUTHORIZED_USER),
        (_credential_class('google.oauth2.service_account')(),
         GCPIdentityType.SERVICE_ACCOUNT),
        (_credential_class('google.auth.compute_engine.credentials')(),
         GCPIdentityType.METADATA_SERVICE_ACCOUNT),
        (_credential_class('google.auth.impersonated_credentials')(),
         GCPIdentityType.IMPERSONATED_SERVICE_ACCOUNT),
        (_credential_class('google.auth.external_account_authorized_user')(),
         GCPIdentityType.EXTERNAL_ACCOUNT),
        (_credential_class('example.credentials')(), GCPIdentityType.UNKNOWN),
    ],
)
def test_adc_credential_type_classification(credential, expected):
    assert GCP._get_identity_type_from_credentials(credential) == expected


def test_external_account_subclass_classification():
    external_base = _credential_class('google.auth.external_account')
    identity_pool_credentials = _credential_class('google.auth.identity_pool',
                                                  bases=(external_base,))

    assert GCP._get_identity_type_from_credentials(
        identity_pool_credentials()) == GCPIdentityType.EXTERNAL_ACCOUNT


@pytest.mark.parametrize(
    ('identity_type', 'can_expire'),
    [
        (GCPIdentityType.AUTHORIZED_USER, True),
        (GCPIdentityType.SERVICE_ACCOUNT, False),
        (GCPIdentityType.METADATA_SERVICE_ACCOUNT, False),
        (GCPIdentityType.EXTERNAL_ACCOUNT, True),
        (GCPIdentityType.IMPERSONATED_SERVICE_ACCOUNT, True),
        (GCPIdentityType.UNKNOWN, True),
    ],
)
def test_adc_credential_expiration_is_conservative(identity_type, can_expire):
    assert identity_type.can_credential_expire() is can_expire


@pytest.mark.parametrize(
    ('credential_module', 'can_expire'),
    [
        ('google.oauth2.credentials', True),
        ('google.oauth2.service_account', False),
        ('example.credentials', True),
    ],
)
def test_cloud_expiration_uses_adc_type(credential_module, can_expire):
    credentials = _credential_class(credential_module)()
    with mock.patch.object(gcp_cloud,
                           '_get_default',
                           return_value=(credentials, 'test-project')):
        assert GCP().can_credential_expire() is can_expire


def test_user_identity_comes_from_adc_not_active_gcloud_account():
    authorized_user_class = _credential_class('google.oauth2.credentials')
    credentials = authorized_user_class()
    credentials.account = 'adc-user@example.com'
    credentials.get_cred_info = lambda: {
        'principal': 'adc-user@example.com',
    }

    with mock.patch.object(gcp_cloud,
                           '_get_default',
                           return_value=(credentials, 'test-project')), \
         mock.patch.object(GCP,
                           '_get_adc_principal',
                           return_value='adc-user@example.com'), \
         mock.patch.object(GCP,
                           'get_project_id',
                           return_value='test-project'), \
         mock.patch.object(GCP,
                           '_get_active_gcloud_account',
                           return_value='different@example.com') as get_gcloud:
        identities = GCP._get_user_identities('workspace-config-adc')

    assert identities == [['adc-user@example.com [project_id=test-project]']]
    get_gcloud.assert_not_called()


def test_metadata_adc_refreshes_to_resolve_service_account():
    metadata_class = _credential_class('google.auth.compute_engine.credentials')
    credentials = metadata_class()
    credentials.service_account_email = 'default'
    credentials.get_cred_info = lambda: {'principal': 'default'}

    def refresh(_):
        credentials.service_account_email = (
            'metadata-sa@test-project.iam.gserviceaccount.com')

    credentials.refresh = refresh

    principal = GCP._get_adc_principal(credentials,
                                       GCPIdentityType.METADATA_SERVICE_ACCOUNT)

    assert principal == 'metadata-sa@test-project.iam.gserviceaccount.com'


def test_external_adc_prefers_impersonated_service_account():
    external_base = _credential_class('google.auth.external_account')
    identity_pool_credentials = _credential_class('google.auth.identity_pool',
                                                  bases=(external_base,))
    credentials = identity_pool_credentials()
    credentials.service_account_email = (
        'target@test-project.iam.gserviceaccount.com')
    credentials._audience = '//iam.googleapis.com/projects/123/providers/p'
    credentials.get_cred_info = lambda: None

    principal = GCP._get_adc_principal(credentials,
                                       GCPIdentityType.EXTERNAL_ACCOUNT)

    assert principal == 'target@test-project.iam.gserviceaccount.com'


def test_direct_external_adc_rejects_shared_provider_audience():
    external_base = _credential_class('google.auth.external_account')
    identity_pool_credentials = _credential_class('google.auth.identity_pool',
                                                  bases=(external_base,))
    credentials = identity_pool_credentials()
    credentials._audience = (
        '//iam.googleapis.com/projects/123/locations/global/'
        'workloadIdentityPools/pool/providers/provider')
    credentials.get_cred_info = lambda: None

    with pytest.raises(exceptions.CloudUserIdentityError,
                       match='direct external-account ADC'):
        GCP._get_adc_principal(credentials, GCPIdentityType.EXTERNAL_ACCOUNT)


def test_unknown_adc_without_principal_is_rejected():
    credentials = _credential_class('example.credentials')()
    credentials.get_cred_info = lambda: None

    with pytest.raises(exceptions.CloudUserIdentityError,
                       match='could not determine the principal'):
        GCP._get_adc_principal(credentials, GCPIdentityType.UNKNOWN)


def test_legacy_user_adc_resolves_identity_with_oauth_userinfo():
    credentials = _credential_class('google.oauth2.credentials')()
    credentials.account = ''
    credentials.get_cred_info = lambda: None
    execute = mock.Mock(return_value={'email': 'legacy@example.com'})
    oauth2 = mock.Mock()
    oauth2.userinfo.return_value.get.return_value.execute = execute

    with mock.patch.object(gcp_cloud.gcp, 'build', return_value=oauth2):
        principal = GCP._get_adc_principal(credentials,
                                           GCPIdentityType.AUTHORIZED_USER)

    assert principal == 'legacy@example.com'
    execute.assert_called_once_with(num_retries=3)


def test_authorized_user_account_hint_must_match_verified_subject():
    credentials = _credential_class('google.oauth2.credentials')()
    credentials.account = 'edited@example.com'
    execute = mock.Mock(return_value={'email': 'actual@example.com'})
    oauth2 = mock.Mock()
    oauth2.userinfo.return_value.get.return_value.execute = execute

    with mock.patch.object(gcp_cloud.gcp, 'build', return_value=oauth2), \
         pytest.raises(exceptions.CloudUserIdentityError,
                       match='does not match the OAuth subject'):
        GCP._get_adc_principal(credentials, GCPIdentityType.AUTHORIZED_USER)


def test_workforce_user_identity_binds_subject_and_audience():
    credentials = _credential_class(
        'google.auth.external_account_authorized_user')()
    credentials.is_user = True
    credentials.audience = '//iam.googleapis.com/locations/global/pools/pool-a'
    execute = mock.Mock(return_value={
        'email': 'worker@example.com',
        'id': 'provider-subject-123',
    })
    oauth2 = mock.Mock()
    oauth2.userinfo.return_value.get.return_value.execute = execute

    with mock.patch.object(gcp_cloud.gcp, 'build', return_value=oauth2):
        principal_a = GCP._get_adc_principal(credentials,
                                             GCPIdentityType.EXTERNAL_ACCOUNT)
        credentials.audience = (
            '//iam.googleapis.com/locations/global/pools/pool-b')
        principal_b = GCP._get_adc_principal(credentials,
                                             GCPIdentityType.EXTERNAL_ACCOUNT)

    assert principal_a.startswith('worker@example.com [workforce_subject=')
    assert principal_a != principal_b


def test_gcloud_and_authorized_user_adc_mismatch_is_rejected():
    with mock.patch.object(gcp_cloud.shutil,
                           'which',
                           return_value='/usr/bin/gcloud'), \
         mock.patch.object(GCP,
                           '_get_identity_type',
                           return_value=GCPIdentityType.AUTHORIZED_USER), \
         mock.patch.object(gcp_cloud.os.path, 'isfile', return_value=True), \
         mock.patch.object(GCP,
                           '_find_application_key_path',
                           return_value='/adc.json'), \
         mock.patch.object(GCP,
                           '_get_active_gcloud_account',
                           return_value='gcloud@example.com'), \
         mock.patch.object(GCP, 'get_project_id', return_value='test-project'), \
         mock.patch.object(
             GCP,
             'get_active_user_identity',
             return_value=[
                 'adc@example.com [project_id=test-project]'
             ]):
        ok, reason = GCP._check_credentials([], [])

    assert not ok
    assert reason is not None
    assert 'authorize as different users' in reason
    assert 'gcloud@example.com' in reason
    assert 'adc@example.com' in reason


def test_check_credentials_uses_custom_cloudsdk_config(tmp_path):
    home = tmp_path / 'home'
    custom_root = home / 'custom-gcloud'
    custom_root.mkdir(parents=True)
    (custom_root / 'access_tokens.db').write_bytes(b'custom token database')
    (custom_root / 'credentials.db').write_bytes(b'custom credential database')
    custom_adc = custom_root / 'application_default_credentials.json'
    custom_adc.write_text('{}', encoding='utf-8')

    credentials = _credential_class('google.oauth2.credentials')()
    crm = mock.Mock()
    crm.projects.return_value.testIamPermissions.return_value.execute.\
        return_value = {
            'permissions': [],
        }
    with mock.patch.dict(
            'os.environ', {
                'HOME': str(home),
                'CLOUDSDK_CONFIG': '~/custom-gcloud',
            }), \
         mock.patch.object(gcp_cloud.shutil,
                           'which',
                           return_value='/usr/bin/gcloud'), \
         mock.patch.object(GCP,
                           '_get_identity_type',
                           return_value=GCPIdentityType.AUTHORIZED_USER), \
         mock.patch.object(GCP,
                           'get_project_id',
                           return_value='test-project'), \
         mock.patch.object(
             GCP,
             'get_active_user_identity',
             return_value=[
                 'active@example.com [project_id=test-project]'
             ]), \
         mock.patch.object(GCP,
                           '_get_active_gcloud_account',
                           return_value='active@example.com'), \
         mock.patch.object(gcp_cloud,
                           '_get_default',
                           return_value=(credentials, 'test-project')), \
         mock.patch.object(gcp_cloud.gcp, 'build', return_value=crm):
        assert GCP._find_application_key_path() == str(custom_adc)
        ok, reason = GCP._check_credentials([], [])

    assert ok
    assert reason is None


def test_workforce_adc_compares_verified_email_with_gcloud():
    workforce_identity = (
        'worker@example.com [workforce_subject=subject-digest] '
        '[project_id=test-project]')
    credentials = _credential_class(
        'google.auth.external_account_authorized_user')()
    crm = mock.Mock()
    crm.projects.return_value.testIamPermissions.return_value.execute.\
        return_value = {
            'permissions': [],
        }
    with mock.patch.object(gcp_cloud.shutil,
                           'which',
                           return_value='/usr/bin/gcloud'), \
         mock.patch.object(GCP,
                           '_get_identity_type',
                           return_value=GCPIdentityType.EXTERNAL_ACCOUNT), \
         mock.patch.object(GCP,
                           'get_project_id',
                           return_value='test-project'), \
         mock.patch.object(GCP,
                           'get_active_user_identity',
                           return_value=[workforce_identity]), \
         mock.patch.object(GCP,
                           '_get_active_gcloud_account',
                           return_value='worker@example.com'), \
         mock.patch.object(gcp_cloud,
                           '_get_default',
                           return_value=(credentials, 'test-project')), \
         mock.patch.object(gcp_cloud.gcp, 'build', return_value=crm):
        ok, reason = GCP._check_credentials([], [])

    assert ok
    assert reason is None


def test_portable_adc_mounts_isolated_gcloud_credential_store():
    authorized_user = _credential_class('google.oauth2.credentials')()
    service_account = _credential_class('google.oauth2.service_account')()
    staged_mounts = {
        '~/.config/gcloud/credentials.db': '/staged/credentials.db',
    }
    with mock.patch.object(GCP,
                           '_find_application_key_path',
                           return_value='/adc.json'), \
         mock.patch.object(GCP,
                           '_get_adc_principal',
                           return_value='adc@example.com'), \
         mock.patch.object(GCP,
                           '_get_active_gcloud_account',
                           return_value='adc@example.com'), \
         mock.patch.object(GCP,
                           '_get_active_gcloud_base_account',
                           return_value='adc@example.com'), \
         mock.patch.object(GCP,
                           '_stage_gcloud_cli_credentials',
                           return_value=staged_mounts):
        with mock.patch.object(gcp_cloud,
                               '_get_default',
                               return_value=(authorized_user, 'project')):
            user_mounts = GCP().get_credential_file_mounts()
        with mock.patch.object(gcp_cloud,
                               '_get_default',
                               return_value=(service_account, 'project')):
            service_account_mounts = GCP().get_credential_file_mounts()

    assert user_mounts == {
        **staged_mounts,
        gcp_cloud.DEFAULT_GCP_APPLICATION_CREDENTIAL_PATH: '/adc.json',
    }
    assert service_account_mounts == {
        **staged_mounts,
        gcp_cloud.DEFAULT_GCP_APPLICATION_CREDENTIAL_PATH: '/adc.json',
    }


def test_credential_mount_collection_fails_closed_when_adc_load_fails():
    with mock.patch.object(gcp_cloud,
                           '_get_default',
                           side_effect=RuntimeError('invalid ADC')), \
         mock.patch.object(GCP, '_stage_gcloud_cli_credentials') as stage:
        assert GCP().get_credential_file_mounts() == {}
    stage.assert_not_called()


def test_staged_gcloud_database_contains_only_active_account(tmp_path):
    source_root = tmp_path / 'gcloud'
    configurations = source_root / 'configurations'
    configurations.mkdir(parents=True)
    (source_root / 'active_config').write_text('default\n', encoding='utf-8')
    (configurations / 'config_default').write_text(
        '[core]\naccount = inactive@example.com\nproject = project\n',
        encoding='utf-8')

    with sqlite3.connect(source_root / 'credentials.db') as database:
        database.execute('CREATE TABLE credentials '
                         '(account_id TEXT PRIMARY KEY, value BLOB)')
        database.executemany('INSERT INTO credentials VALUES (?, ?)', [
            ('active@example.com', b'active-secret'),
            ('inactive@example.com', b'inactive-secret'),
        ])
    with sqlite3.connect(source_root / 'access_tokens.db') as database:
        database.execute(
            'CREATE TABLE access_tokens '
            '(account_id TEXT PRIMARY KEY, access_token TEXT, '
            'token_expiry TIMESTAMP, rapt_token TEXT, id_token TEXT)')
        database.executemany(
            'INSERT INTO access_tokens VALUES (?, ?, ?, ?, ?)', [
                ('active@example.com', 'active-token', None, None, None),
                ('inactive@example.com', 'inactive-token', None, None, None),
            ])

    with mock.patch.object(GCP,
                           '_get_configured_gcloud_impersonation',
                           return_value=None):
        mounts = GCP._stage_gcloud_cli_credentials('active@example.com',
                                                   'active@example.com',
                                                   source_root=str(source_root),
                                                   staging_root=str(tmp_path /
                                                                    'staged'))

    staged_credentials = mounts['~/.config/gcloud/credentials.db']
    with sqlite3.connect(staged_credentials) as database:
        rows = database.execute(
            'SELECT account_id, value FROM credentials').fetchall()
    assert rows == [('active@example.com', b'active-secret')]

    staged_tokens = mounts['~/.config/gcloud/access_tokens.db']
    with sqlite3.connect(staged_tokens) as database:
        token_accounts = database.execute(
            'SELECT account_id FROM access_tokens').fetchall()
    assert token_accounts == [('active@example.com',)]

    parser = configparser.RawConfigParser()
    parser.read(mounts['~/.config/gcloud/configurations'] + '/config_default')
    assert parser.get('core', 'account') == 'active@example.com'
    staged_active_config = mounts['~/.config/gcloud/active_config']
    with open(staged_active_config, encoding='utf-8') as file:
        assert file.read() == 'default'


def test_staged_gcloud_credentials_use_custom_cloudsdk_config(tmp_path):
    account = 'active@example.com'
    home = tmp_path / 'home'
    default_root = home / '.config' / 'gcloud'
    custom_root = home / 'custom-gcloud'

    for root, secret, token, project in [
        (default_root, b'default-secret', 'default-token', 'default-project'),
        (custom_root, b'custom-secret', 'custom-token', 'custom-project'),
    ]:
        configurations = root / 'configurations'
        configurations.mkdir(parents=True)
        (root / 'active_config').write_text('default', encoding='utf-8')
        (configurations / 'config_default').write_text(
            f'[core]\naccount = {account}\nproject = {project}\n',
            encoding='utf-8')
        with sqlite3.connect(root / 'credentials.db') as database:
            database.execute('CREATE TABLE credentials '
                             '(account_id TEXT PRIMARY KEY, value BLOB)')
            database.execute('INSERT INTO credentials VALUES (?, ?)',
                             (account, secret))
        with sqlite3.connect(root / 'access_tokens.db') as database:
            database.execute(
                'CREATE TABLE access_tokens '
                '(account_id TEXT PRIMARY KEY, access_token TEXT, '
                'token_expiry TIMESTAMP, rapt_token TEXT, id_token TEXT)')
            database.execute('INSERT INTO access_tokens VALUES (?, ?, ?, ?, ?)',
                             (account, token, None, None, None))

    with mock.patch.dict(
            'os.environ', {
                'HOME': str(home),
                'CLOUDSDK_CONFIG': '~/custom-gcloud',
            }), \
         mock.patch.object(GCP,
                           '_get_configured_gcloud_impersonation',
                           return_value=None):
        mounts = GCP._stage_gcloud_cli_credentials(account,
                                                   account,
                                                   staging_root=str(tmp_path /
                                                                    'staged'))

    assert set(mounts) == {
        '~/.config/gcloud/configurations',
        '~/.config/gcloud/active_config',
        '~/.config/gcloud/credentials.db',
        '~/.config/gcloud/access_tokens.db',
    }
    with sqlite3.connect(mounts['~/.config/gcloud/credentials.db']) as database:
        credential_rows = database.execute(
            'SELECT account_id, value FROM credentials').fetchall()
    assert credential_rows == [(account, b'custom-secret')]
    with sqlite3.connect(
            mounts['~/.config/gcloud/access_tokens.db']) as database:
        token_rows = database.execute(
            'SELECT account_id, access_token FROM access_tokens').fetchall()
    assert token_rows == [(account, 'custom-token')]

    parser = configparser.RawConfigParser()
    parser.read(mounts['~/.config/gcloud/configurations'] + '/config_default')
    assert parser.get('core', 'project') == 'custom-project'


def test_staged_gcloud_credentials_honor_active_config_env(tmp_path):
    account = 'active@example.com'
    source_root = tmp_path / 'gcloud'
    configurations = source_root / 'configurations'
    configurations.mkdir(parents=True)
    (source_root / 'active_config').write_text('config-a', encoding='utf-8')
    (configurations / 'config_config-a').write_text(
        f'[core]\naccount = {account}\nproject = project-a\n', encoding='utf-8')
    (configurations / 'config_config-b').write_text(
        f'[core]\naccount = {account}\nproject = project-b\n', encoding='utf-8')
    with sqlite3.connect(source_root / 'credentials.db') as database:
        database.execute('CREATE TABLE credentials '
                         '(account_id TEXT PRIMARY KEY, value BLOB)')
        database.execute('INSERT INTO credentials VALUES (?, ?)',
                         (account, b'active-secret'))

    with mock.patch.dict('os.environ',
                         {'CLOUDSDK_ACTIVE_CONFIG_NAME': 'config-b'}), \
         mock.patch.object(GCP,
                           '_get_configured_gcloud_impersonation',
                           return_value=None):
        mounts = GCP._stage_gcloud_cli_credentials(account,
                                                   account,
                                                   source_root=str(source_root),
                                                   staging_root=str(tmp_path /
                                                                    'staged'))

    staged_configurations = mounts['~/.config/gcloud/configurations']
    assert not os.path.exists(
        os.path.join(staged_configurations, 'config_config-a'))
    staged_config = os.path.join(staged_configurations, 'config_config-b')
    assert os.path.isfile(staged_config)
    parser = configparser.RawConfigParser()
    parser.read(staged_config)
    assert parser.get('core', 'project') == 'project-b'
    with open(mounts['~/.config/gcloud/active_config'],
              encoding='utf-8') as file:
        assert file.read() == 'config-b'


@pytest.mark.parametrize('active_config', ['../unsafe', 'missing'])
def test_staged_gcloud_credentials_reject_invalid_active_config(
        tmp_path, active_config):
    source_root = tmp_path / 'gcloud'
    configurations = source_root / 'configurations'
    configurations.mkdir(parents=True)
    (source_root / 'active_config').write_text('default', encoding='utf-8')
    (configurations / 'config_default').write_text(
        '[core]\naccount = active@example.com\n', encoding='utf-8')

    with mock.patch.dict('os.environ',
                         {'CLOUDSDK_ACTIVE_CONFIG_NAME': active_config}), \
         pytest.raises(exceptions.CloudUserIdentityError):
        GCP._stage_gcloud_cli_credentials('active@example.com',
                                          'active@example.com',
                                          source_root=str(source_root),
                                          staging_root=str(tmp_path / 'staged'))
    assert not (tmp_path / 'staged').exists()


@pytest.mark.parametrize('invalid_source', ['', 'missing', 'not-a-directory'])
def test_staged_gcloud_credentials_reject_invalid_cloudsdk_config(
        tmp_path, invalid_source):
    source = '' if not invalid_source else str(tmp_path / invalid_source)
    if invalid_source == 'not-a-directory':
        (tmp_path / invalid_source).write_text('not a directory',
                                               encoding='utf-8')

    with mock.patch.dict('os.environ', {'CLOUDSDK_CONFIG': source}), \
         pytest.raises(exceptions.CloudUserIdentityError,
                       match='CLOUDSDK_CONFIG'):
        GCP._stage_gcloud_cli_credentials('active@example.com',
                                          'active@example.com',
                                          staging_root=str(tmp_path / 'staged'))


def test_staged_gcloud_credentials_require_portable_database(tmp_path):
    source_root = tmp_path / 'gcloud'
    configurations = source_root / 'configurations'
    configurations.mkdir(parents=True)
    (source_root / 'active_config').write_text('default', encoding='utf-8')
    (configurations / 'config_default').write_text(
        '[core]\naccount = active@example.com\n', encoding='utf-8')

    with mock.patch.object(GCP,
                           '_get_configured_gcloud_impersonation',
                           return_value=None), \
         pytest.raises(exceptions.CloudUserIdentityError,
                       match='portable credential database'):
        GCP._stage_gcloud_cli_credentials('active@example.com',
                                          'active@example.com',
                                          source_root=str(source_root),
                                          staging_root=str(tmp_path / 'staged'))


def test_impersonated_adc_exposes_verified_legacy_source_identity():
    impersonated = _credential_class('google.auth.impersonated_credentials')()
    source = _credential_class('google.oauth2.credentials')()
    impersonated._source_credentials = source

    def get_principal(credentials, _):
        if credentials is impersonated:
            return 'target@test-project.iam.gserviceaccount.com'
        assert credentials is source
        return 'source@example.com'

    with mock.patch.object(gcp_cloud,
                           '_get_default',
                           return_value=(impersonated, 'test-project')), \
         mock.patch.object(GCP,
                           '_get_adc_principal',
                           side_effect=get_principal), \
         mock.patch.object(GCP,
                           'get_project_id',
                           return_value='test-project'):
        identities = GCP._get_user_identities(
            'workspace-config-legacy-impersonation')

    assert identities == [[
        'target@test-project.iam.gserviceaccount.com '
        '[project_id=test-project]'
    ], ['source@example.com [project_id=test-project]']]


def test_impersonated_credential_expiration_follows_source():
    impersonated = _credential_class('google.auth.impersonated_credentials')()
    metadata = _credential_class('google.auth.compute_engine.credentials')()
    user = _credential_class('google.oauth2.credentials')()

    impersonated._source_credentials = metadata
    assert not GCP._credentials_can_expire(impersonated)

    impersonated._source_credentials = user
    assert GCP._credentials_can_expire(impersonated)


def test_legacy_identity_enum_values_are_preserved():
    assert GCPIdentityType('') == GCPIdentityType.AUTHORIZED_USER
    assert (GCPIdentityType('iam.gserviceaccount.com') ==
            GCPIdentityType.SERVICE_ACCOUNT)
