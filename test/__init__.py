#!/usr/bin/env python
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file The project's unit tests entrypoint and manager.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2021-present Rodrigo Siqueira
import importlib
import unittest
import sys
import os

modules = []
path = os.path.dirname(__file__)
sys.path.insert(0, path)
os.chdir(path)

# To run all tests, we must search for all test modules on the directory. All files
# with the extension ".py" is considered a test module in this directory.
for fname in os.listdir(path):
    if fname.endswith(".py") and not fname.startswith('__init__'):
        spec = importlib.util.spec_from_file_location(fname[:-3], fname)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)

# Loads all test cases present on this directory.
# @since 0.1.1
def load_tests(loader, *_):
    tests = [loader.loadTestsFromModule(module) for module in modules]
    return unittest.TestSuite(tests)
