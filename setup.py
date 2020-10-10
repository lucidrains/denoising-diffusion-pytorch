from setuptools import setup, find_packages

setup(
  name = 'denoising-diffusion-pytorch',
  packages = find_packages(),
  version = '0.5.2',
  license='MIT',
  description = 'Denoising Diffusion Probabilistic Models - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/denoising-diffusion-pytorch',
  keywords = [
    'artificial intelligence',
    'generative models'
  ],
  install_requires=[
    'einops',
    'numpy',
    'pillow',
    'torch',
    'torchvision',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)