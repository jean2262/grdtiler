from setuptools import setup, find_packages

setup(
    name='grdtiler',
    use_scm_version={'write_to': 'grdtiler/_version.py'},
    setup_requires=['setuptools_scm'],
    description='A package for tilling GRD products',
    author='jean2262',
    author_email='jean-renaud.miadana@oceanscope.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pytest',
        'pytest-cov',
        'tqdm',
        'shapely',
        'xarray',
        'xsar',
        'xarray-safe-s1',
        'xradarsat2',
        'xarray-safe-rcm',
        'xsarsea',
    ],
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
