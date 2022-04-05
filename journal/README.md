# Lab journal

Things to try next:
- [X] lrelu activation on conv layers (works quite well)
- [X] larger latent size 64 -> 128 (no significant change)
- [X] Sigmoid on output layer (didn't work)
- [X] No activation on decoder output layer (works quite well)
- [X] Try increasing the learning rate to 5e-05 (works well)
- [ ] Try increasing the learning rate to 1e-04
- [ ] Pad images instead of stretching them to the target size
- [ ] Try only the 'standing' tag to constrain the domain to fewer poses
- [ ] Use kernel size of 5 on conv layers (some promising preliminary results, needs better checking)
- [ ] Use `he_normal` kernel initialisation on conv layers
- [ ] Linear activation on decoder output layer

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

