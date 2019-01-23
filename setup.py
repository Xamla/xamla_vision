

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    name='xamla_vision',
    version='0.0.1',
    description='set of functions doing various vision tasks in python',
    long_description=open('README.md').read(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        ],
    keywords='ROSVITA robotics python3 xamla',
    author='Alexandre Stueben',
    author_email='alexandre.stueben@xamla.com',
    packages=['xamla_vision'],
    package_dir={'': 'src'},
    url='https://github.com/Xamla/xamla_vision',
    license='LICENSE',
    install_requires=[
        'numpy >= 1.11.0',
        'asyncio >= 3.4.3',
        'rospkg >= 1.1.4',
        'catkin_pkg >= 0.4.5',
        'rospy >= 1.12.13'
    ],
)

setup(**setup_args)
