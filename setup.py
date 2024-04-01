import pathlib
from setuptools import setup, find_packages


def test_suite():
    import unittest

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    return test_suite


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="aiidalab_qe_hp",
    version="0.0.1",
    description="AiiDAlab quantum ESPRESSO app plugin for HP code",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/superstart54/aiidalab_qe_hp",
    author="Xing Wang",
    author_email="xingwang1991@gmail.com",
    license="MIT License",
    classifiers=[],
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "aiida.workflows": [
            "aiidalab_qe_hp = aiidalab_qe_hp.workflows",
        ],
        "aiidalab_qe.properties": [
            "hp = aiidalab_qe_hp:hp",
        ],
    },
    package_data={},
    python_requires=">=3.9",
    test_suite="setup.test_suite",
)
