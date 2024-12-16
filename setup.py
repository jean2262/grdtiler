from setuptools import setup, find_packages

setup(
    name='grdtiler',
    version='0.0.3',
    description='A package for tilling GRD products',
    author='jean2262',
    author_email='jrenaud495@gmail.com',
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
    python_requires='>=3.11',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
