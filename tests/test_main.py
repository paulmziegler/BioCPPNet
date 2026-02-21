from click.testing import CliRunner

from src.main import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main)
    assert result.exit_code == 0
    assert "Initializing BioCPPNet Pipeline..." in result.output
