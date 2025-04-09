from unittest.mock import patch, MagicMock

import pytest

from jointly_hic.entrypoint import setup_logging, main


def test_setup_logging(mocker):
    """Test setup_logging."""
    # Mock logging.getLogger
    get_logger_patch = mocker.patch("jointly_hic.entrypoint.logging.getLogger")
    # Mock logging.FileHandler
    file_handler_patch = mocker.patch("jointly_hic.entrypoint.logging.FileHandler")
    # Mock logging.StreamHandler
    mocker.patch("jointly_hic.entrypoint.logging.StreamHandler")
    # Mock logging.Formatter
    formatter_patch = mocker.patch("jointly_hic.entrypoint.logging.Formatter")

    # Create some CLI args
    args = mocker.Mock()
    args.log = "test_log"
    args.log_level = "INFO"

    # Test setup_logging
    setup_logging(args)

    # Test
    get_logger_patch.assert_called_once_with("joint_pca")
    get_logger_patch.return_value.handlers.clear.assert_called_once()
    get_logger_patch.return_value.setLevel.assert_called_once_with(args.log_level)
    file_handler_patch.assert_called_once_with(
        args.log,
        mode="w",
        encoding="utf-8",
    )
    file_handler_patch.return_value.setLevel.assert_called_once_with(args.log_level)
    formatter_patch.assert_called_once_with("%(asctime)s::%(levelname)s::%(funcName)s:   %(message)s")
    file_handler_patch.return_value.setFormatter.assert_called_once_with(formatter_patch.return_value)


def test_main_success():
    """Test the main function when it completes successfully."""
    with (
        patch("jointly_hic.entrypoint.JointlyCommandLineInterface") as mock_cli,
        patch("jointly_hic.entrypoint.setup_logging") as mock_setup_logging,
    ):
        # Mock the parser to simulate command line arguments being parsed successfully
        mock_parser = MagicMock()
        mock_cli.return_value.parser = mock_parser
        mock_parser.parse_args.return_value = MagicMock(func=lambda x: None)

        # Mock the logger to capture logging without sending output anywhere
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        # Run the main function
        main()

        # Check that logging was called with the expected messages
        mock_logger.info.assert_any_call("Starting jointly-hic")
        mock_logger.info.assert_any_call("Finished jointly-hic")


def test_main_exception():
    """Test the main function when an exception is raised."""
    with (
        patch("jointly_hic.entrypoint.JointlyCommandLineInterface") as mock_cli,
        patch("jointly_hic.entrypoint.setup_logging") as mock_setup_logging,
    ):
        # Mock the parser to simulate command line arguments being parsed successfully
        mock_parser = MagicMock()
        mock_cli.return_value.parser = mock_parser
        error = Exception("Test error")
        mock_parser.parse_args.return_value = MagicMock(func=lambda x: (_ for _ in ()).throw(error))

        # Mock the logger to capture logging without sending output anywhere
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        # Run the main function and expect it to raise an exception
        with pytest.raises(Exception) as exc_info:
            main()

        # Check that the exception was logged
        mock_logger.exception.assert_called_with(error)

        # Verify the exception message
        assert str(exc_info.value) == "Test error", "Unexpected exception message"
