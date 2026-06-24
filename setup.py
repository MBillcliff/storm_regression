from setuptools import setup, find_packages

setup(
    name='storm_regression',
    version='0.1',
    description='Regression-based geomagnetic storm forecasting project',
    author='Matthew Billcliff',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
)