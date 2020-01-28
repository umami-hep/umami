import unittest  # noqa
from unittest import mock
import argparse
import os
from umami.preprocessing import GetParser


class PreprocessingTestParser(unittest.TestCase):
    """
    Test the implementation of the Prerocessing command line parser.
    """
    config_file = os.path.join(os.path.dirname(__file__),
                               "test_preprocess_config.yaml")

    def setUp(self):
        self.config_file = os.path.join(os.path.dirname(__file__),
                                        "test_preprocess_config.yaml")

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(config_file=config_file,
                                                undersampling=False,
                                                scaling=True,
                                                apply_scales=False,
                                                prepare_large=True,
                                                var_dict="test", tracks=False
                                                ))
    def test_Parser(self, mock_args):
        parser = GetParser()
        self.assertEqual(parser.config_file, self.config_file)
        self.assertFalse(parser.tracks)
        # self.assertIsNone(parser.cut_config_file)

   @mock.patch('argparse.ArgumentParser.parse_args',
   return_value=argparse.Namespace(config_file=config_file,
                                               tracks=True))
    def test_ParserTracks(self, mock_args):
       self.parser = GetParser()
       self.assertTrue(self.parser.tracks)

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(config_file=config_file,
                                                undersampling=True))
    def test_ParserTracks(self, mock_args):
        self.parser = GetParser()
        self.assertTrue(self.parser.tracks)
