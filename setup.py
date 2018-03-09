from distutils.core import setup
setup(
  name = 'boundmixofgaussians',
  packages = ['boundmixofgaussians'],
  version = '1.0',
  description = ' find a bound on the global maximum of a mixture of Gaussians. Assumes equal covariance functions for now',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/lionfish0/boundmixofgaussians.git',
  download_url = 'https://github.com/lionfish0/boundmixofgaussians.git',
  keywords = ['Gaussian','bounds','Mixture of Gaussians'],
  classifiers = [],
  install_requires=['numpy'],
)
