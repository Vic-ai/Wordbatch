#!/usr/bin/env python
import os
import platform

import numpy
from Cython.Distutils import build_ext
from setuptools import Extension, setup

# Detect architecture
is_arm = platform.machine() in ('arm64', 'aarch64')
is_macos = platform.system() == 'Darwin'

if os.name == 'nt':
	extra_compile_args = ["/openmp", "/Ox", "/arch:AVX2", "/fp:fast"]
	extra_link_args = []
elif is_arm:
	# ARM architecture (Apple Silicon, etc.) - no AVX2, use NEON-friendly flags
	# Note: don't use -std=gnu11 as MurmurHash3.cpp is C++
	extra_compile_args = ["-O3", "-ffast-math", "-ftree-vectorize"]
	extra_link_args = []
	if is_macos:
		# macOS with Homebrew's libomp
		extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
		extra_link_args += ["-lomp"]
else:
	# x86_64 Linux/macOS
	# Note: don't use -std=gnu11 as MurmurHash3.cpp is C++
	extra_compile_args = ["-O3", "-fopenmp", "-ffast-math", "-mavx2", "-ftree-vectorize"]
	extra_link_args = ["-fopenmp"]
	if is_macos:
		# macOS x86 with Homebrew's libomp
		extra_compile_args = ["-O3", "-Xpreprocessor", "-fopenmp", "-ffast-math", "-mavx2", "-ftree-vectorize"]
		extra_link_args = ["-lomp"]

setup(
	name='Wordbatch',
	version='2.1.0',
	description='Python library for distributed AI processing pipelines, using swappable scheduler backends',
	url='https://github.com/anttttti/Wordbatch',
	author='Antti Puurula',
	author_email='antti.puurula@yahoo.com',
	packages=['wordbatch',
			  'wordbatch.pipelines',
			  'wordbatch.extractors',
			  'wordbatch.models',
			  'wordbatch.transformers'
	],
	license='GNU GPL 2.0',
	classifiers=[
		"Development Status :: 4 - Beta",
		"Intended Audience :: Developers",
		"License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Programming Language :: Cython",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Software Development :: Libraries :: Python Modules",
	],
	install_requires=['Cython>=0.29.20', 'scikit-learn', 'python-Levenshtein', 'lz4', 'randomgen==1.21.2',
					  'numpy>=1.23.2,<2.0', 'scipy>=1.9.0,<2.0', 'pandas>=1.5.0,<3.0', 'wheel>=0.33.4'],
	extras_require={'dev': ['nltk', 'textblob', 'keras', 'pyspark', 'dask', 'distributed', 'ray']},

	cmdclass= {'build_ext': build_ext},
	ext_modules= [
				  Extension("wordbatch.extractors.extractors",
							["wordbatch/extractors/extractors.pyx", "wordbatch/extractors/MurmurHash3.cpp"],
							libraries= [],
							include_dirs=[numpy.get_include(), '.'],
							extra_compile_args = extra_compile_args,
							extra_link_args=extra_link_args),
				  Extension("wordbatch.models.ftrl",
							["wordbatch/models/ftrl.pyx"],
							libraries=[],
							include_dirs=[numpy.get_include(), '.'],
							extra_compile_args=extra_compile_args,
							extra_link_args=extra_link_args),
				  Extension("wordbatch.models.ftrl32",
							["wordbatch/models/ftrl32.pyx"],
							libraries=[],
							include_dirs=[numpy.get_include(), '.'],
							extra_compile_args=extra_compile_args,
							extra_link_args=extra_link_args),
				  Extension("wordbatch.models.fm_ftrl",
							["wordbatch/models/fm_ftrl.pyx", "wordbatch/models/avx_ext.c"],
							libraries= [],
							include_dirs=[numpy.get_include(), '.'],
							extra_compile_args = extra_compile_args,
							extra_link_args=extra_link_args),
				  Extension("wordbatch.models.nn_relu_h1",
							["wordbatch/models/nn_relu_h1.pyx"],
							libraries= [],
							include_dirs=[numpy.get_include(), '.'],
							extra_compile_args = extra_compile_args,
							extra_link_args=extra_link_args),
				  Extension("wordbatch.models.nn_relu_h2",
							["wordbatch/models/nn_relu_h2.pyx"],
							libraries= [],
							include_dirs=[numpy.get_include(), '.'],
							extra_compile_args = extra_compile_args,
							extra_link_args=extra_link_args),
		]
)
