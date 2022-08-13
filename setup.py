import os
from setuptools import setup


def _get_version():
    with open('jax_utils/__init__.py') as fp:
        for line in fp:
            if line.startswith('__version__'):
                g = {}
                exec(line, g)  # pylint: disable=exec-used
                return g['__version__']
        raise ValueError(
            '`__version__` not defined in `jax_utils/__init__.py`')


def _parse_requirements(requirements_txt_path: os.path.join(os.path.dirname(__file__), "requirements.txt")):
    with open(requirements_txt_path) as fp:
        return tuple(filter(lambda line: line.lstrip()[0] in {"#", '-'}, fp.read().splitlines()))


setup(
    name='jax_utils',
    version=_get_version(),
    packages=('jax_utils',),
    install_requires=_parse_requirements(),
)
