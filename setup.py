from setuptools import find_packages, setup


setup(
    name='giggle',
    packages=find_packages(include=['giggle']),
    version='0.0.1',
    description='Library for massive discs and GI wiggles',
    author='Cristiano Longarini',
    license='lol',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)