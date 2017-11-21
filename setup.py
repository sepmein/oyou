from setuptools import setup

setup(name='oyou',
      version='0.1',
      description='Build and make use of tensorflow model with pleasure and flexibility',
      url='http://github.com/sepmein/oyou',
      author='Spencer Zhang',
      author_email='sepmein@hotmail.com',
      license='MIT',
      packages=['oyou'],
      zip_safe=False,
      install_requires=[
          'tensorflow'
      ],
      test_suite='nose.collector',
      tests_require=['nose'])