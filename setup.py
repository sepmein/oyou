from setuptools import setup

setup(name='oyou',
      version='0.4.0',
      description='Build and make use of tensorflow model with pleasure and flexibility',
      url='http://github.com/sepmein/oyou',
      author='Spencer Zhang',
      author_email='sepmein@hotmail.com',
      keywords='tensorflow machine-learning graph-util',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
      ],
      license='MIT',
      packages=['oyou'],
      zip_safe=False,
      install_requires=[
          'tensorflow',
          'numpy'
      ],
      test_suite='nose.collector',
      tests_require=['nose'])
