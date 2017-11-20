from setuptools import setup, find_packages

# Load text for description and license
with open('README.md') as f:
    readme = f.read()
with open('LICENSE') as f:
    license = f.read()

# Go!
setup(
    # Module name (lowercase)
    name='pints',
    # Remember to keep this in sync with pints/__init__.py
    version='0.0.1',
    description='Probabilistic Inference in Noisy Time-Series',
    long_description=readme,
    license=license,
    #author='',
    #author_email='',
    url='https://github.com/martinjrobins/pints',
    # Packages to include
    packages=find_packages(include=('pints', 'pints.*')),
)
