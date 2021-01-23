#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file The project's unit tests entrypoint and manager.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2021-present Rodrigo Siqueira
import sys
import os

# Setting the current directory as the Python's runtime working directory. This
# allows us to use relative paths on our test files.
path = os.path.dirname(__file__)
sys.path.insert(0, path)
os.chdir(path)
