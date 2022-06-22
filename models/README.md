# The Vector-quantized Variational Autoencoder
The vector-quantized variational autoencoder was introduced by [Van Oord et al.](https://arxiv.org/abs/1711.00937) to
solve a particular problem concerning the 'standard' implementation of the variational auto-encoder by
[Kingma et al.](https://arxiv.org/abs/1312.6114). The original variational auto-encoder demonstrated an auto-encoder
with a very interesting property: the latent 'codes' to reconstruct outputs were modeled as multivariate Gaussian
densities, allowing 'tweakable' implementations to generate unseen reconstructions both close to the average, and far
from them.

The particular problem mentioned above, however, was that standard VAEs did not scale up to larger samples. When run on
MNIST digits, for example, VAEs worked fine. The trouble with MNIST, of course, is that it is pretty close to a toy
dataset. There's hardly any practical domain where 28x28 greyscale images are useful. Instead, what we use in real-life
are high-resolution. It turned out that the VAE did quite poorly on them.

My earlier attempts showed that this was indeed the case for the FAFA image collection as well. Standard VAEs showed
clear studio backgrounds that were consistent across the dataset, but it produce very, very blurry patches for those
parts of the image that actually showed the model as the main subject of the image. This means that the VAE suffered
from a phenomenon known as "posterior collapse", the area where the VAE was supposed to shine: in capturing the
variability of the dataset and everything in between. To be able to interpolate _between_ the images in the collection.

In came the VQ-VAE by Van Oord et al. It solved the "posterior collapse" but it came at a price. This is where things
get interesting of course - the parts they don't stress in the paper because it distracts the reader from the qualities
or, if you like, 'selling points' of the model. But here it is, after several months of studying, I find myself slowly
but steadily learning the gritty details of the VQ-VAE and its quirks. The main problem points:

- Where the 'reparametrization trick' of the VAE is already quite a difficult subject matter to grasp, wait until you've
  seen the two-stage encoder/quantizer/decoder + PixelCNN parts of _this_ system.
- The VQ-VAE is considerably more resource-intensive to use than the VAE. It's still very much possible to use without
  Nvidia A100 behemoths or TPU monsters, but training and hyperparamter-optimizing this VQ-VAE beast is quite
  challenging.
