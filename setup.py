#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['kornia==0.5.8',
    'medpy==0.4.0',
    'monai==0.6.0',
    'nibabel==3.2.1',
    'pytorch_lightning==1.3.8',
    'torch==1.10.2',
    'torchio==0.18.47',
    'torchvision==0.11.3',
    "flask"]

test_requirements = [ ]

dependency_links = ['https://download.pytorch.org/whl/cu111']

setup(
    author="Nathan Decaux",
    author_email='nathan.decaux@imt-atlantique.fr',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Label propagation using deep registration",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='labelprop',
    name='deep-labelprop',
    packages=find_packages(include=['labelprop', 'labelprop.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nathandecaux/labelprop',
    version='0.1.0',
    zip_safe=False,
)
