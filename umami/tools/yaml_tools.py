"""
code taken from
https://github.com/Tristan-Sweeney-CambridgeConsultants/ccorp_yaml_include
and adapted for the needs of the umami framework
"""
import os
import types

import ruamel.yaml
import ruamel.yaml.composer
import ruamel.yaml.constructor
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode


class CompositingComposer(ruamel.yaml.composer.Composer):
    compositors = {k: {} for k in (ScalarNode, MappingNode, SequenceNode)}

    @classmethod
    def add_compositor(cls, tag, compositor, *, nodeTypes=(ScalarNode,)):
        for nodeType in nodeTypes:
            cls.compositors[nodeType][tag] = compositor

    @classmethod
    def get_compositor(cls, tag, nodeType):
        return cls.compositors[nodeType].get(tag, None)

    def __compose_dispatch(self, anchor, nodeType, callback):
        event = self.parser.peek_event()
        compositor = self.get_compositor(event.tag, nodeType) or callback
        if isinstance(compositor, types.MethodType):
            return compositor(anchor)
        return compositor(self, anchor)

    def compose_scalar_node(self, anchor):
        return self.__compose_dispatch(anchor, ScalarNode, super().compose_scalar_node)

    def compose_sequence_node(self, anchor):
        return self.__compose_dispatch(
            anchor, SequenceNode, super().compose_sequence_node
        )

    def compose_mapping_node(self, anchor):
        return self.__compose_dispatch(
            anchor, MappingNode, super().compose_mapping_node
        )


class ExcludingConstructor(ruamel.yaml.constructor.Constructor):
    filters = {k: [] for k in (MappingNode, SequenceNode)}

    @classmethod
    def add_filter(cls, filter_yaml, *, nodeTypes=(MappingNode,)):
        for nodeType in nodeTypes:
            cls.filters[nodeType].append(filter_yaml)

    def construct_mapping(self, node):  # pylint: disable=arguments-differ
        node.value = [
            (key_node, value_node)
            for key_node, value_node in node.value
            if not any(f(key_node, value_node) for f in self.filters[MappingNode])
        ]
        return super().construct_mapping(node)

    def construct_sequence(self, node):  # pylint: disable=arguments-differ
        node.value = [
            value_node
            for value_node in node.value
            if not any(f(value_node) for f in self.filters[SequenceNode])
        ]
        return super().construct_sequence(node)


class YAML(ruamel.yaml.YAML):
    def __init__(self, *args, **kwargs):
        if "typ" not in kwargs:
            kwargs["typ"] = "safe"
        if isinstance(kwargs["typ"], list) and len(kwargs["typ"]) == 1:
            kwargs["typ"] = kwargs["typ"][0]
        elif kwargs["typ"] not in ["safe", "unsafe"]:
            raise Exception(
                f"Can't do typ={kwargs['typ']} parsing w/ composition timedirectives!"
            )

        if "pure" not in kwargs:
            kwargs["pure"] = True
        elif not kwargs["pure"]:
            raise Exception(
                "Can't do non-pure python parsing w/ composition time directives!"
            )

        super().__init__(*args, **kwargs)
        self.Composer = CompositingComposer
        self.Constructor = ExcludingConstructor

    def compose(self, stream):
        """
        at this point you either have the non-pure Parser (which has its own reader and
        scanner) or you have the pure Parser.
        If the pure Parser is set, then set the Reader and Scanner, if not already set.
        If either the Scanner or Reader are set, you cannot use the non-pure Parser,
            so reset it to the pure parser and set the Reader resp. Scanner if necessary
        """
        _, parser = self.get_constructor_parser(stream)
        try:
            return self.composer.get_single_node()
        finally:
            parser.dispose()
            try:
                self._reader.reset_reader()
            except AttributeError:
                pass
            try:
                self._scanner.reset_scanner()
            except AttributeError:
                pass

    def fork(self):
        yml = type(self)(typ=self.typ, pure=self.pure)
        yml.composer.anchors = self.composer.anchors
        return yml


def include_compositor(self, anchor):  # pylint: disable=unused-argument
    event = self.parser.get_event()
    yml = self.loader.fork()
    path = os.path.join(os.path.dirname(self.loader.reader.name), event.value)
    with open(path) as f:
        return yml.compose(f)


def exclude_filter(key_node, value_node=None):
    value_node = value_node or key_node  # copy ref if None
    return key_node.tag == "!exclude" or value_node.tag == "!exclude"


CompositingComposer.add_compositor("!include", include_compositor)
ExcludingConstructor.add_filter(exclude_filter, nodeTypes=(MappingNode, SequenceNode))


if __name__ == "__main__":
    import argparse
    import pprint

    yaml = YAML(typ="safe", pure=True)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file")

    parse_args = argparser.parse_args()

    with open(parse_args.file) as yaml_file:
        pprint.pprint(yaml.load(yaml_file))
