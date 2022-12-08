from setuptools import setup

path_requirements = 'requirements.txt'
list_packages = [
    'batch_bayeso',
]

with open(path_requirements) as f:
    required = f.read().splitlines()

setup(
    name='batch-bayeso',
    version='0.1.0',
    author='Jungtaek Kim',
    author_email='jtkim@postech.ac.kr',
    url='https://bayeso.org',
    license='MIT',
    description='Batch Bayesian optimization with BayesO',
    packages=list_packages,
    python_requires='>=3.6, <4',
    install_requires=required,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
