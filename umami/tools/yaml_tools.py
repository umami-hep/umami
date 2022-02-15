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
    """Composer class extension of ruamel."""

    compositors = {k: {} for k in (ScalarNode, MappingNode, SequenceNode)}

    @classmethod
    def add_compositor(cls, tag, compositor, *, node_types=(ScalarNode,)):
        """adding compositor

        Parameters
        ----------
        tag : object
            tag
        compositor : object
            compositor
        node_types : tuple, optional
            node type, by default (ScalarNode,)
        """
        for node_type in node_types:
            cls.compositors[node_type][tag] = compositor

    @classmethod
    def get_compositor(cls, tag, node_type):
        """compositor getter.

        Parameters
        ----------
        tag : object
            tag
        node_type : object
            node type

        Returns
        -------
        compositor object
            compositors
        """
        return cls.compositors[node_type].get(tag, None)

    def __compose_dispatch(self, anchor, node_type, callback):
        """Compose dispatch.

        Parameters
        ----------
        anchor : object
            anchor
        node_type : object
            node type
        callback : function
            callback

        Returns
        -------
        compositor object
            compositor
        """
        event = self.parser.peek_event()
        compositor = self.get_compositor(event.tag, node_type) or callback
        if isinstance(compositor, types.MethodType):
            return compositor(anchor)
        return compositor(self, anchor)

    def compose_scalar_node(self, anchor):
        """Compose scalar node.

        Parameters
        ----------
        anchor : object
            anchor

        Returns
        -------
        compose_dispatch object
            __compose_dispatch
        """
        return self.__compose_dispatch(anchor, ScalarNode, super().compose_scalar_node)

    def compose_sequence_node(self, anchor):
        """Compose sequence node.

        Parameters
        ----------
        anchor : object
            anchor

        Returns
        -------
        compose_dispatch object
            __compose_dispatch from super class
        """
        return self.__compose_dispatch(
            anchor, SequenceNode, super().compose_sequence_node
        )

    def compose_mapping_node(self, anchor):
        """Compose mapping node.

        Parameters
        ----------
        anchor : object
            anchor

        Returns
        -------
        compose_dispatch object
            __compose_dispatch from super class
        """
        return self.__compose_dispatch(
            anchor, MappingNode, super().compose_mapping_node
        )


class ExcludingConstructor(ruamel.yaml.constructor.Constructor):
    """Constructor class extension of ruamel."""

    filters = {k: [] for k in (MappingNode, SequenceNode)}

    @classmethod
    def add_filter(cls, filter_yaml, *, node_types=(MappingNode,)):
        """Adding filter.

        Parameters
        ----------
        filter_yaml : object
            filter
        node_types : tuple, optional
            node types, by default (MappingNode,)
        """
        for nodeType in node_types:
            cls.filters[nodeType].append(filter_yaml)

    def construct_mapping(self, node):  # pylint: disable=arguments-differ
        """Construct mapping.

        Parameters
        ----------
        node : object
            node

        Returns
        -------
        construct_mapping
            construct_mapping from super class
        """
        node.value = [
            (key_node, value_node)
            for key_node, value_node in node.value
            if not any(f(key_node, value_node) for f in self.filters[MappingNode])
        ]
        return super().construct_mapping(node)

    def construct_sequence(self, node):  # pylint: disable=arguments-differ
        """Construct sequence.

        Parameters
        ----------
        node : object
            node

        Returns
        -------
        construct_sequence object
            construct_sequence from super class
        """
        node.value = [
            value_node
            for value_node in node.value
            if not any(f(value_node) for f in self.filters[SequenceNode])
        ]
        return super().construct_sequence(node)


class YAML(ruamel.yaml.YAML):
    """Yaml interpreter adding !include option which support anchors."""

    def __init__(self, *args, **kwargs):
        """Initialise and check arguments.

        Parameters
        ----------
        *args : args
            args
        **kwargs : kwargs
            kwargs

        Raises
        ------
        Exception
            if type not parsable
        Exception
            if non-pure python passing
        """
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
        """at this point you either have the non-pure Parser (which has its own reader
        and scanner) or you have the pure Parser.
        If the pure Parser is set, then set the Reader and Scanner, if not already set.
        If either the Scanner or Reader are set, you cannot use the non-pure Parser,
        so reset it to the pure parser and set the Reader resp. Scanner if necessary

        Parameters
        ----------
        stream : stream
            stream

        Returns
        -------
        node
            node
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
        """Fork function.

        Returns
        -------
        yaml
            yaml object
        """
        yml = type(self)(typ=self.typ, pure=self.pure)
        yml.composer.anchors = self.composer.anchors
        return yml


def include_compositor(self, anchor):  # pylint: disable=unused-argument
    """Compositor inclusion function

    Parameters
    ----------
    self : object
        self constructor from class
    anchor : object
        anchor

    Returns
    -------
    yml.compose
        composed object
    """
    event = self.parser.get_event()
    yml = self.loader.fork()
    path = os.path.join(os.path.dirname(self.loader.reader.name), event.value)
    with open(path) as y_file:
        return yml.compose(y_file)


def exclude_filter(key_node, value_node=None):
    """Filder exclusion.

    Parameters
    ----------
    key_node : object
        key node
    value_node : object, optional
        value node, by default None

    Returns
    -------
    bool
        filter
    """
    value_node = value_node or key_node  # copy ref if None
    return key_node.tag == "!exclude" or value_node.tag == "!exclude"


CompositingComposer.add_compositor("!include", include_compositor)
ExcludingConstructor.add_filter(exclude_filter, node_types=(MappingNode, SequenceNode))


if __name__ == "__main__":
    import argparse
    import pprint

    yaml = YAML(typ="safe", pure=True)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file")

    parse_args = argparser.parse_args()

    with open(parse_args.file) as yaml_file:
        pprint.pprint(yaml.load(yaml_file))
