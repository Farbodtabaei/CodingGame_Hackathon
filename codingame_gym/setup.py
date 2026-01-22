from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='codingame_gym',
    version='0.1',
    packages=find_packages(),
    description='A python class to instantiate Codingame games as OpenAI Gym environments.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Fabio Zinno',
    author_email='fzinno@ea',
    url='https://gitlab.ea.com/a-team/ml-development/codingame_gym',
    install_requires=required,
)
