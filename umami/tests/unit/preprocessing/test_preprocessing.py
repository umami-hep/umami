"""Unit test if preprocessing."""
import argparse
import os
import unittest  # noqa
from unittest import mock

from umami.configuration import logger, set_log_level
from umami.preprocessing import get_parser

set_log_level(logger, "DEBUG")


class PreprocessingTestParser(unittest.TestCase):
    """
    Test the implementation of the Prerocessing command line parser.
    """

    config_file = os.path.join(os.path.dirname(__file__), "test_preprocess_config.yaml")

    def setUp(self):
        self.config_file = os.path.join(
            os.path.dirname(__file__), "test_preprocess_config.yaml"
        )

    @mock.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(
            config_file=config_file,
            prepare=False,
            undersampling=False,
            scaling=True,
            apply_scales=False,
            write=False,
            var_dict="test",
            tracks=False,
        ),
    )
    def test_parser(self, mock_args):  # pylint: disable=W0613
        """Test parser

        Parameters
        ----------
        mock_args : mock_args
            passed arguments from command line via mock
        """
        parser = get_parser()
        self.assertEqual(parser.config_file, self.config_file)
        self.assertFalse(parser.tracks)


class PreprocessingTestGetScaleDict(unittest.TestCase):
    """
    Test the implementation of the GetScaleDict function.
    """

    config_file = os.path.join(os.path.dirname(__file__), "test_preprocess_config.yaml")

    def setUp(self):
        self.config_file = os.path.join(
            os.path.dirname(__file__), "test_preprocess_config.yaml"
        )

    # TODO: write test
