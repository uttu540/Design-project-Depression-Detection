from setuptools import setup, find_packages

setup(
    name='Depression_Detection',
    include_package_data=True,
    install_requires=[
        'flask',
    ],
    packages=find_packages()
)