# Lab journal

Things to try next:
"Standard" VAE
- [X] lrelu activation on conv layers (works quite well)
- [X] larger latent size 64 -> 128 (no significant change)
- [X] Sigmoid on output layer (didn't work)
- [X] No activation on decoder output layer (works quite well)
- [X] Try increasing the learning rate to 5e-05 (works well)
- [X] Try increasing the learning rate to 1e-04 (no significant change)
- [X] Pad images instead of stretching them to the target size (works quite well)
- [X] Use `he_normal` kernel initialisation on conv layers (works quite well)
- [X] Try only the 'standing' tag to constrain the domain to fewer poses (improvements, still blurry)

VQ-VAE
- [X] Use simpler feature scaling to floats in range [0..1] to aid in reconstruction simplification (is fine)
- [X] Implement vector-quantized VAE (excellent, huge improvement)
- [ ] Implement the pixelCNN
- [x] Tweak learning rate (worked well)
- [ ] Resume training on a saved model
- [ ] Implement the vqvae model training as an Argo Workflow 
- [ ] Refactor reconstruction callback so that it can write directly to the data bucket
- [ ] Refactor checkpoint callback so that it can write directly to the data bucket
- [ ] Tweak the latent size, how does it affect the two loss components?
- [ ] Use kernel size of 3 or 5 on conv layers (some promising preliminary results, needs better checking)
- [ ] Linear activation on decoder output layer

## 2022-04-21
I spent about two days refactoring code to end up basically exactly where I was three days ago. I spent time redesigning
the data loader, creating an implementation that used queues from the multiprocessing standard library, only to find out
that Keras already implements data loaders that way under the hood, if you fit your training data with 
`use_multiprocessing=True`. So that was helpful ... in the learning process at least.

So in the end the only functional change I made was to implement the loader as a `keras.utils.data_utils.Sequence` which
is for the best I guess, although it is decidely more complex than the simple generator function I implemented earlier.
But whatever. At least I got a new run started today, with a slightly higher learning rate, to see how it fares.

Couple of hours later it is already apparent that the higher learning rate works quite well, the model does benefit from
it so it appears. Let's see where the model performance ends up after 128 epochs.

## 2022-04-15
After try 1 the VQVAE already shows much, much better reconstructions than the standard VAE. The 256 epochs I train for,
show much clearer figures. Interestingly, the model focuses much more on the general picture than on trivial things,
such as logos and watermarks. It shows decent quality images and it can certainly improve with a bit more training, for
sure. This is all on a 4-layer conv mirrored to deconv on the decoder, with the largest layer having 64 filters. It'd
definitely be worth adding another 64 filter, or even a 128 filter layer.

Thing is, though, that I'm still pretty clueless on the role of the pixelCNN. Until I know what it does, I'm not sure if
it's worth training deeper VAE nets. 

## 2022-04-14
I've successfully refactored the autoencoder model to a vector-quantized VAE but it took me about a day to structure it
to my implementation. The changes are very, very complex, and that is saying something when starting out from a VAE to
begin with. The concept of the VQVAE 'codebook' in itself is very complex, although it is maybe best described as an
one-hot embedding space (hence the 'quantized' predicate) that encodes the input feature space in a lower-dimensional
latent space. 

With the new implementation, of course now my previous loss figures are utterly meaningless and I have to inspect the
reconstructed images to see how it fares. One interesting thing is that the number of weights on the VQ-VAE is much,
much lower than on the vanilla VAE. The VAE uses a flattening operation and a subsequent fully connected layer close to
the latent, as the latent is a simple vector. The fully connected layer, however, was absolutely humongous, given the
size of the images on the one hand, and the size of the latent vector on the other. At least that is solved in the
VQVAE implementation, but I may have to add in some conv/deconv layers to give the model enough weights to learn a good
reconstruction.

I haven't implemented the pixelCNN yet, however, and I'm still not exactly clear on its purpose. I suspect it has to do
with enhancing the reconstructions from the code book, but why it isn't trained end-to-end but as a separate part of a
pipeline I cannot fathom. Perhaps it has to do with a certain incompatibility with the quantization part of the training
of the encoder, quantizer and decoder part. Also, I have to reach the part on how to sample from the codebook to 
generate novel images.

Interesting, after the first half hour, the model outputs blocky reconstructions of stick figures. The output is much
less fuzzy than on the first iterations of the VAE, but also much more abstract.

## 2022-04-13
Looks I've run into something
that [others have as well](https://www.reddit.com/r/deeplearning/comments/rjpsmt/what_are_some_variational_autoencoder/)
. 'Standard' VAEs appear to scale up badly to higher-resolution images. With that, almost anything beyond MNIST digits.
640x640 appears to be solidly beyond the resolution that VAEs can handle. Fortunately, there is a 
[code example](https://keras.io/examples/generative/vq_vae/) from 
Keras that makes refactoring my model easier. That sample is designed to work on the dreaded MNIST digits again, but 
using the larger encoder and decoder layers, it should not be a lot of work to refactor the model.

## 2022-04-12
The bucket is working very nicely, it's an excellent solution. It allows me to power off the T4 machine whilst still
having full access to the artifacts.

The 'standing' constraint works fine on reducing the loss on the reconstructions, but the result is still too blurry for
my taste. I'm going to switch tack and make a few changes. First of all, I'm going to refactor the model to be closer to
the original VAE implementation, with inputs scaled between 0 and 1 and using bce loss for the reconstructions. I intend
to increase the batch size a bit, in the hopes that that will reduce the jaggedness of the loss curve. It jumps around
a bit too much, which I suspect inhibits the learning process. So I refactored the data generator function to scale the
images to within [0 + epsilon ... 1 - epsilon]. I factored in the epsilons, as I don't want the model to go to great
lengths trying to reach exactly 0. and 1., because I put the sigmoid activation back in on the last decoder layer. I 
didn't need to touch the reconstruction sampler callback because it automagically scales the image pixel values back up
again. I also used the binary cross-entropy loss for the reconstructions to get back to the original implementation as
well. I haven't looked into that loss function yet, but I wonder if BCE will help reduce the infamous 'haze' on MSE VAE
generated images.

Learned a lesson today on lab journal logging today as well. Setting up de `gcsfuse` command on my GPU machine is far
from trivial. I still use it to mount the bucket in order to directly write reconstruction sample images to it, and
model checkpoints, but I forgot the command. So here goes:
```shell
sudo GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json gcsfuse \
  -o allow_other \
  --implicit-dirs \
  --uid 1001 --gid 1001 \
  antfield ~/mnt/antfield
```
It needs sudo because of -o allow_other, and it needs the google credentials path because root doesn't know where to 
look.

## 2022-04-11
This weekend's session ran to 927 of the 1024 intended epochs until the disk was full. I already created a bucket to
store the data in, but 927 epochs is fine to get a clearer picture of what the model is capable of, running for a longer
period. These 927 epochs took about 34 hours. The model learned _some_ extra domain features, but it still does not
produce clear pictures of humans. Strangely enough, the KL loss was still _rising_ after 900 epochs, and the
reconstruction loss still dropping! Do I have to configure a higher learning rate after all? Or is this normal? Anyway,
I do start to think that the domain is a little bit too diverse or complicated for the model. I may have to constrain it
to the `standing` tagged images.

What did work well, was to initialize the conv kernels with `he_normal`ly distributed weights. It clearly shows about
25% reduce in reconstruction loss after 100 epochs. At least on the run I did this weekend, I don't have the means to
create statistically significant measurements on my artist's free tier budget.

So, the reconstruction MSE loss ended up on ~1.5e4 after 900 epochs. That's pretty OK, but quite frankly it's not good
enough to reconstruct life-like pictures. I'm going to have to take a different approach and I'm going to start with the
`standing` tag constraint.

Started a new run with only standing images, to see whether the model size is the bottleneck here. If it manages to
produce convincing standing poses, then that could be enough for a working system for now. Then, I can always try to
enlargen the model to accomodate a larger selection of poses. I also moved writing artifacts to a storage bucket, to
prevent flooding the harddisk once more. I'm leaving the training data on the harddisk however, because it needs to do
so many reading operations from disk. I don't believe that either the model or my wallet is going to benefit from having
the training data in a storage bucket. Already I can see a substantial loss improvement over the full pose range. 
Although I'd have to inspect the images soon in order to be sure.

## 2022-04-08
I refactored the image loader so that it produces padded instead of stretched image tensors, but now the model produces
unintelligible reconstruction garbage. Clearly I made some miscalculation in scaling/unscaling the input/output images.
Previously I just divided the input space by 255 to map the pixel range to floats between 0 and 1, but this, besides
producing a spectacular drop in loss, resulted in no usable output images. Back to the drawing board. 

I took a look at
the [`standardize`](https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/image_data_generator.py#L707)
method from Keras and took their method of sample-wise normalization to apply to my images. At least this produced
intelligible reconstructions from previous runs. Phew, at least the epoch callback function produces FAFA-like images
again! The loss is still significantly lower than on the standard Keras scaling image preprocessor functions, the gains
in quality cannot be explained away just by having more "blank" image data from the padding. The padding is quite
substantial, of course: it is (640 - 427) / 640 = 33 % of the total data. So the loss should be at least 33% lower, as
the reconstruction MSE loss over the padding should be 0. However, the KL loss drops from 670 to 419 at epoch 10, the
reconstruction loss from 8.9e4 to 3.7e4 on otherwise identical settings.

I'm letting this one run for a full 256 epochs and see where it ends up. I found the bug that produced the
unintelligible reconstruction results: the reconstruction image sampler used the old data generator! I still would like
to decouple the padding from the scaling operation though, and see how a sample-wise default normalization pans out.
This indicates that the spectacular drops in losses from the /255 scaler could still be meaningful!

Results are in for the 256 epochs. The model still learns after 256 epochs, so it definitely is not at the end of its
use of tuning the model weights. Question is where to go from here. I'm thinking of adding code to resume training on a
saved model. This way, I can alternate days - one is for tweaking the model, the other day is for training continuation
and seeing how many epochs it takes to finish training the model.

One thing I noticed today, is that after about 50 epochs or so, the Kullback-Leibler divergence loss starts going up,
just very slightly but decidedly so. At about a constant rate, not sure how much but still. I'm unsure on what this
actually means, but it might mean that the model needs the latent vector as part of the network, to learn the features
at a very condensed state. Is the latent size too small? I would have hoped not, at a size of 64 I already think it's
rather large. Tweaking 64 numbers to generate images is quite a bit of a hassle. So, I may have to experiment a little
on changing the latent size and see how it affects the loss figures, both on the reconstruction loss and the KL loss.

## 2022-04-06
Tried increasing the learning rate from 5e-5 to 1e-4 this morning but it did not appear to have any impact on the
reconstruction loss development. The initial loss spike on the KL-loss is somewhat lower, but it settles on the same
value as a learning rate of 5e-5. The 1e-4 loss curve of the reconstruction loss hugs the 5e-5 curve closely, so I might
just as well stick to 5e-5.

Also implemented a image tensor padding data generator, that was fun. It's a bit unfortunate that the ImageDataGenerator
class by Keras is so inaccessible, it's a deeply nested entangled set of classes and methods that all require 
re-implementing. Instead, I just implemented a generator function of a couple of lines, without the overload of 22 
function parameters. A lot more manageable. I took a bit of a shortcut in normalizing the image data. I just divide by
255, leaving tensors with floats between 0 and 1. I believe this should be fine, I'd be very surprised if the network
did not know how to handle it. Admittedly, this data isn't centered, but the bias weights in the layers should easily be
able to handle data within [0., 1.].

The upside of having a padding data generator is twofold:
- it does not require the network to learn 'stretched' representations of the domain
- it drops the requirement of having to scale the image to original size on generated/reconstructed images.

## 2022-04-05
The T4 experiments of yesterday were quite spectacular. The reconstruction loss at the end of the full 256 epoch 
training session ended up at about 5.8e4 which is better than any run before, and resulted in clear human-like figure
reconstructions! There are clear lighting and shadow effects, some poses do not show up very clearly but the current
architecture is definitely on the right track. I'll start increasing the learning rate a little because the end of the
session definitely did not flatten on the loss, so there's more to be gained from the net. This time I'm _only_ going to
fiddle with the learning rate.

Started a new run and it definitely benefits from the higher learning rate of 5e-5. The model goes through the 1e5
reconstruction on epoch 7 instead of epoch 22 on the same batch size. At epoch 9, it is 30 epochs ahead of the previous
best run. End of the day result is looking quite spectacular, both loss-wise and visual inspection. Figures in the
reconstructed images start more and more resembling actual human shapes and hues. Loss on the latest run dropped from 
~5.8e4 to ~4.1e4, so that's a big improvement. Again: after 256 epochs the loss curve is still pointing downwards a few
degrees, so training longer will quite probably result in better results. First I'm going to try a run with a 1e-4 
learning rate tomorrow and see where that ends up.

I also spent some time refactoring the data generator. I'm going to try and see if I can build a generator that pads the
input images instead of stretching them to the target size. Frankly, I'm a bit surprised that this option isn't provided
for the ImageDataGenerator class - it isn't supported. If I can find the time, I may spend some to try and make a pull
request for this.

Trying the new data preprocessor for a first run, the effect is quite dramatic on the loss figures. The KL loss is about
half from the standard ImageDataGenerator. The reconstruction loss is an order of magnitude smaller - shrinking from
7.4e4 to 4.2e4 after one hour of training on otherwise identical settings. Visual inspection, though, is not quite as
impressive - reconstructions of the human shapes are just as vague. Hopefully though, results will improve otherwise the
loss does not confer much information on the actual reconstruction quality. There is of course the matter of 
normalisation. I'm having the `save_image` export the image, but it automatically scales the values to values within
[0..255]. I'm guessing it does this by de-centering and de-scaling, but I'm not sure. Instead, I could just hard-scale
the network output by `min(255, (max(0, output) * 255)` to undo the normalisation. In the case of the auto-encoder, it
is fully unsupervised - the normalized inputs are supposed to be the same as the outputs, that's where the loss function
gets its score from.

## 2022-04-04
Despite my intentions to the contrary, I tried some changes simultaneously today. But disabling a specific activation
on the last layer paid off big. The reconstruction error breached the barrier of 1e5 today, at a KL loss that is still
on the large side however, but the loss improvements have been substantial today. The main thing that dropping `tanh` 
activation on the last layer is that the model figures take on a much more realistic color, they were very dark shapes
previously. The unspecified activation (is this linear by default?) results in much better skin-toned model "ghosts",
right from epoch 2.

I also switched to T4 instances, which greatly speeds up training, compared to my own GTX 1060 laptop. This is the 
second set of changes I made: I switched from a batch size of 8 to 10 and increased the first conv layer of 16 filters
to 32 to see if that results in better model performance.

## 2022-04-03
I'm trying out too many changes at once, this hampers inspection on what actually works better. Today I'm trying to
change _only_ the activation on the decoder output layer, switching from sigmoid to tanh. Taking one step at a time.
Trouble is of course, that I'm impatient. I got some decent results earlier, but I need to test out one change at a time
to see whether it actually improves the loss. So: yesterday's run ended on a reconstruction loss of about 3.2e5 and a
KL loss of about 400. They all settle on about that KL number, regardless of what the network looks like. 

Run 2022-04-03_11h51m10s uses a somewhat larger field of view (kernel size 5) on the conv layers, just like the run I
did yesterday, but then only with the activation function on the decoder output set to hyperbolic tangent. After 16
minutes it's already at the 56 minute mark performance of my best run so far, from friday. This should shape up to be a
new best run for sure. Strangely enough, the KL loss starts out much higher on this one than the friday run.

Also: it's about time that I switch from my untrusty old laptop to a cloud GPU/TPU server, my sessions keep getting
killed by Python somwhere after 150-200 epochs, due to the size of the model script. It takes over 22 Gb of (virtual)
memory, I don't quite know why yet. At a batch size of 8, a single batch currently holds 8 images of 640x640x3 float32's
so at that could certainly not account for more than just a couple of megabytes. The image metadata dataframe is
probably substantially larger, in JSON format it's already ~120Mb. But I sure don't know where the other gigabytes go.

I also made sure to copy all the Python model files and config.yaml to the artifacts folder to archive not only the
model, checkpoints and reconstructions, but also the input files that generated it.

## 2022-04-01
Started out this week from 
the [Keras example code for variational autoencoders](https://keras.io/examples/generative/vae/). I departed from that
setup quite quickly, as it does not appear to transfer very well to larger rgb images. Biggest improvement I got, apart
from tweaking the convolutional layer sizes and stacking, was swapping from a sigmoid activation to a tanh activation on
the last decoder conv2dtranspose layer and swapping a binary cross-entropy reconstruction loss for a mean squared error
reconstruction loss. The BCE loss spiralled out of control very quickly and produced _only_ completely black images. I'm
not quite sure why this is the case - many convolutional VAE architectures appear to successfully use sigmoid activation
in tandem with BCE loss. This is a big question: why does it not work in this case?

Found this thread on BCE/MSE VAE reconstruction loss: https://github.com/pytorch/examples/issues/399. Maybe I switch to
pytorch and see if the network behaves any differently from keras/tensorflow.

As for the results, I've seen a lot of improvement in the visuals for the network over the week. It produces human-like
ghostly figures, but nothing recognizable yet. The watermarks in some images are a nuisance: if I get rid of the sources
with them, I have a "Ghost" VAE!

