# Multiple Sequence Alignment test package file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018 Rodrigo Siqueira
import sys
import os

__all__ = []
path = os.path.dirname(__file__)
sys.path.insert(0, path)

# We must search for all modules compiled in the directory. All files with
# extension ".so" is considered a module in our package.
for fname in os.listdir(path):
    if fname.endswith(".so"):
        __all__.append(fname[:-3])

# Now that we have told Python about all modules available in the directory,
# it can import all of them at once without any other concerns.
from . import *

# Parses a list of files and produces a list of database entries.
# @param list filenames The list of files to parse.
# @param str ext Parse all files using this parser.
# @return list List containing all entries parsed from files.
def parse(*filenames, **kwargs):
    return parser.any(*filenames, **kwargs)

del fname
del path
del sys
del os
