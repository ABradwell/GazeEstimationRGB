import pathlib
from setuptools import setup, find_packages


def _get_long_description():
    path = pathlib.Path(__file__).parent / 'README.md'
    with open(path, encoding='utf-8') as f:
        long_description = f.read()
    return long_description


def _get_requirements(path):
    with open(path) as f:
        data = f.readlines()
    return data


setup(
    name='webcam_tracker',
    version='0.0.1',
    author='Aiden Bradwell',
    install_requires=_get_requirements('requirements.txt'),
    packages=find_packages(exclude=('tests', )),
    include_package_data=True,
    description='Gaze estimation using hysts/mpiifacegazse through webcams',
)
