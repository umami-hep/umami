import yaml
import os
import warnings
from umami.tools import yaml_loader


class Configuration(object):
    """docstring for Configuration."""

    def __init__(self, yaml_config=None):
        super(Configuration, self).__init__()
        self.yaml_config = yaml_config
        self.yaml_default_config = "configs/preprocessing_default_config.yaml"
        self.LoadConfigFiles()
        self.GetConfiguration()

    def LoadConfigFiles(self):
        self.yaml_default_config = os.path.join(os.path.dirname(__file__),
                                                self.yaml_default_config)
        with open(self.yaml_default_config, "r") as conf:
            self.default_config = yaml.load(conf, Loader=yaml_loader)
        print("Using config file", self.yaml_config)
        with open(self.yaml_config, "r") as conf:
            self.config = yaml.load(conf, Loader=yaml_loader)

    def GetConfiguration(self):
        for elem in self.default_config:
            if elem in self.config:
                if type(self.config[elem]) is dict and "f_" in elem:
                    if 'file' not in self.config[elem]:
                        raise KeyError("You need to specify the 'file' for"
                                       f"{elem} in your config file!")
                    if self.config[elem]['file'] is None:
                        raise KeyError("You need to specify the 'file' for"
                                       f" {elem} in your config file!")
                    if 'path' in self.config[elem]:
                        setattr(self, elem,
                                os.path.join(self.config[elem]['path'],
                                             self.config[elem]['file'])
                                )
                    else:
                        setattr(self, elem, self.config[elem]['file'])

                else:
                    setattr(self, elem, self.config[elem])
            elif self.default_config[elem] is None:
                raise KeyError(f"You need to specify {elem} in your"
                               "config file!")
            else:
                warnings.warn(f"setting {elem} to default value "
                              f"{self.default_config[elem]}")
                setattr(self, elem, self.default_config[elem])

    def GetFileName(self, iteration=None, option=None, extension=".h5"):
        if option is None and iteration is None:
            return self.outfile_name
        out_file = self.outfile_name
        idx = out_file.index(".h5")

        if iteration is None:
            inserttxt = f"-{option}"
        else:
            inserttxt = f"-{option}-file-{iteration:.0f}"\
                        f"_{self.iterations:.0f}"

        out_file = out_file[:idx] + inserttxt + extension
        return out_file
