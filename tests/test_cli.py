
from typer.testing import CliRunner
from optiverse.cli.app import app

def test_version():
    r = CliRunner().invoke(app, ['version'])
    assert r.exit_code == 0
    assert r.stdout.strip()
