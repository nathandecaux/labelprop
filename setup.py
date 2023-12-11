#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages


requirements = [
'Flask>=2.1.0',
'kornia>=0.6.12',
'monai>=1.1',
'nibabel>=3.2.1',
'numpy',
'plotext>=4.2.0',
'lightning']

test_requirements = [ ]

dependency_links = []
with open('README.md') as f:
    readme = f.read()
setup(
    author="Nathan Decaux",
    author_email='nathan.decaux@imt-atlantique.fr',
    python_requires='>=3.8',
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
    entry_points = {
        'console_scripts': [
            'labelprop = labelprop.cli:cli',                  
        ],              
    },
    description="Label propagation using deep registration",
    long_description = readme,
    long_description_content_type = 'text/markdown',
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords='labelprop',
    name='deep-labelprop',
    packages=find_packages(include=['labelprop', 'labelprop.*']),
    package_data={'': ['conf.json']},
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nathandecaux/labelprop',
    version='1.2',
    zip_safe=False,
)
