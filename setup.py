#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
'Flask==2.1.0',
'kornia==0.6.5',
'MedPy==0.4.0',
'monai==0.8.1',
'nibabel==3.2.1',
'numpy==1.20.3',
'plotext==4.2.0',
'pytorch_lightning==1.6.3',
'setuptools==59.5.0',
'torchio==0.18.47']

test_requirements = [ ]

dependency_links = []

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
