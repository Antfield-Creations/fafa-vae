# FAFA-VAE
This repository contains the following:
- A Vector-quantized variational auto-encoder implementation, fully type-hinted, flake8-checked.
- A [scraper module](scraper/scraper.py) to get images from FAFA
- A [config file](config.yaml) being its own Kubernetes manifest
- A [helm chart](charts) containing the Kubernetes operator for the VQ-VAE implementation 
- A test suite to validate various parts of the implementation
- A GitHub Actions workflow to run the test on GitHub 
- A full [lab journal](journal/README.md) containing descriptions of almost every run I did and what my thoughts and doubts were 

Oof, that's a lot. It's because I spent a _lot_ of my free time on this. Let's start with the _really_ complicated part.

## The VQ-VAE
This subject deserves its own README. An elaborate description is in [models/README.md](models/README.md)
