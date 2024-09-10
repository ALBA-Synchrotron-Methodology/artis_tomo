#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom argparser Class

@author: joton
"""

import argparse
import sys
from ...version import __version__
import textwrap
from artis_tomo.utils.config import configFnUserDft


class MultilineArgDefaultHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom Help formatter allowing new lines."""

    def countIndentSpaces(self, line):
        """Count prefix spaces."""
        p = 0
        for c in line:
            if c.isspace():
                p += 1
            else:
                break
        return p

    def removeCommonIdentSpaces(self, lines, firstline=1):
        """Remove prefix spaces given by first spaced line."""
        for line in lines[firstline:]:
            p = self.countIndentSpaces(line)
            if p > 0:
                break

        for k in range(firstline, len(lines)):
            lines[k] = lines[k][p:]

        return lines

    def unwrapText(self, lines):
        """Unwrap lines keeping newlines and indented lines."""
        linesU = [lines[0]]
        previousIsnl = False  # Previous is new line?

        for k in range(1, len(lines)):
            line = lines[k]
            ln = len(line)
            spn = self.countIndentSpaces(line)

            if ln == 0 or ln == spn:
                linesU.append('')
                previousIsnl = True

            elif previousIsnl or spn > 0:
                linesU.append(line)
                previousIsnl = False

            else:
                linesU[-1] += ' ' + line

        return linesU

    def _fill_text(self, text, width, indent):
        text = text.splitlines()              # Make a list of lines
        if len(text[0]) == 0:  # Remove first line if it's empty.
            text.pop(0)
            text = self.removeCommonIdentSpaces(text, 0)
        else:
            text = self.removeCommonIdentSpaces(text)
        text = self.unwrapText(text)
        text = [textwrap.fill(line, width) for line in text]  # Wrap each line
        text = [textwrap.indent(line, indent) for line in text]  # Apply any requested indent.
        text = "\n".join(text)                # Join the lines again
        return text


class ArgumentParser(argparse.ArgumentParser):
    """
    Custom class with default help and version parameters set in last position.

    Section names are in Capital case.
    """

    def __init__(self, *args, **kwargs):
        kwargs['add_help'] = False
        kwargs['formatter_class'] = MultilineArgDefaultHelpFormatter

        # argument_default does not keep default value for store_<true/false>
        # We modify print_help() instead.
        # kwargs['argument_default'] = argparse.argparse.SUPPRESS

        super().__init__(*args, **kwargs)

    def _add_others_group(self):
        others = super(ArgumentParser, self).add_argument_group('Others')
        others.add_argument('-h', '--help', action='help',
                            default=argparse.SUPPRESS,
                            help='Show this help message and exit')
        others.add_argument('--version', action='version', version=__version__,
                            help="Show program's version number and exit")
        # We use action "version" to print configfile description with no need
        # of defining a specific action.
        # We hide it from help list.
        others.add_argument('--cfg_file', action='version',
                            help=argparse.SUPPRESS,
            version=f"""
            Some xpytools programs can read default parameters
            from config files. There are 2 types of config files, for GLOBAL and
            USER environments.
            
            If a parameter is defined in both config files, the one from USER
            environment is considered.
            
            - Global config file is defined by setting environment variable
            XPY_CFG_GLOBAL, i.e. "export XPY_CFG_GLOBAL=/local/opt/xpytools/xpytools.conf".
            
            - User config file is, by default, {configFnUserDft}. It is ignored if
            environment variable XPY_CFG_USER is defined, i.e. "export XPY_CFG_USER=~/configs/xpytools.conf".
            
            """)

    def parse_args(self, *args, **kwargs):
        """Customize actions parameters before parsing."""
        # Required can be set to other parameters.
        requirements = {}
        for action in self._actions:
            option = action.option_strings[0]
            requirements[option] = []

        for action in self._actions:
            option = action.option_strings[0]
            if isinstance(action.required, str):
                requiredForParams = action.required.split(',')
                actionReq = True
                for param in requiredForParams:
                    requirements[param].append(option)
                    actionReq = actionReq and param in sys.argv
                action.required = actionReq
        for action in self._actions:
            option = action.option_strings[0]
            nReq = len(requirements[option])
            if nReq > 0:
                string = ','.join(requirements[option])
                action.help += f' [requires {string}]'

        # Add last "Others" group
        self._add_others_group()
        return super(ArgumentParser, self).parse_args(*args, **kwargs)

    def format_usage(self):
        """Usage label Capitalized."""
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups, prefix='Usage: ')
        return formatter.format_help()

    def format_help(self):
        """Usage label Capitalized."""
        formatter = self._get_formatter()

        # usage
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups, prefix='Usage: ')

        # description
        formatter.add_text(self.description)

        # positionals, optionals and user-defined groups
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()

        # epilog
        formatter.add_text(self.epilog)

        # determine help from format above
        return formatter.format_help()

    def print_help(self, file=None):
        """Supress None and boolean defaults before printing."""
        for action in self._actions:
            if action.default is None or isinstance(action.default, bool):
                action.default = argparse.SUPPRESS

        super(ArgumentParser, self).print_help(file)
