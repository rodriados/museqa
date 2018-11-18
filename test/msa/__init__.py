# Multiple Sequence Alignment test package file.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2018 Rodrigo Siqueira
import os

__all__ = []
path = os.path.dirname(__file__)

# We must search for all modules compiled in the directory. All files with
# extension ".so" is considered a module in our package.
for file in os.listdir(path):
    if file.endswith(".so"):
        __all__.append(file[:-3])

# Now that we have told Python about all modules available in the directory,
# it can import all of them at once without any other concerns.
from . import *

del path
del file
del os