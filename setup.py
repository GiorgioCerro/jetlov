# This file is part of jetron by M. J. Ardiles-Diaz and G. Cerro

from __future__ import print_function
import sys
from setuptools import setup, find_packages

if sys.version_info < (3,9):
    print("jetron requires Python 3.9 or later", file=sys.stderr)
    sys.exit(1)

with open('README.md') as f:
    long_desc = f.read()


setup(name= "jetlov",
      version = '0.0.1',
      description = "Enhancing Jet Tree Tagging through Neural Network Learning of Optimal LundNet Variables",
      author = "G. Cerro",
      author_email = "g.cerro@soton.ac.uk",
      url="https://github.com/GiorgioCerro/jetlov",
      long_description = long_desc,
      #entry_points = {'console_scripts':
      #                ['lundnet = lundnet.scripts.lundnet:main']},
      package_dir = {'': 'src'},
      packages = find_packages('src'),
      zip_safe = False,
      classifiers=[
            'Operating System :: Unix',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            ],
      )
