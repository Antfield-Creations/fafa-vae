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
- [X] Tweak learning rate (no significant improvement, more erratic learning curve)
- [X] Drop the 'standing' filter
- [X] Resume training on a saved model
- [X] Implement the vq-vae model training as an Argo Workflow
- [X] Refactor checkpoint callback so that it can write directly to the data bucket
- [X] Implement the pixelCNN
- [X] Refactor reconstruction callback so that it can write directly to the data bucket
- [X] Tweak the latent size, how does it affect the two loss components?
- [X] Move pixelcnn sampler into separate callback and class
- [ ] Implement `get_config` method for custom vq_vae and pixelcnn models
- [X] Pixelcnn reconstruction callback writing directly to the data bucket
- [ ] Use kernel size of 3 or 5 on conv layers (some promising preliminary results, needs better checking)

## 2022-06-24
Left the windows to my "lab" open in tilted position last night, when a spectacularly loud rainstorm passed over. The 
storm dumped a few centimeters of rain in a couple of short bursts, and I expected to find a water ballet, but 
fortunately there was little to no leakage.

My attempts to "marry" the encoder, decoder and embedding sizes have not worked out particularly well, unfortunately. I
have considerable trouble regaining the sweet spot I had found a couple of days ago, in run 2022-06-13_09h01m09s. The
configuration I tried yesterday ended up collapsing spectacularly over the run a couple of times, in the dreaded "loss
spikes". I included an image to document this:

!["Loss spikes"](images/Screenshot%202022-06-24%20at%2009-16-24%20TensorBoard%20loss%20spikes.png)

I'm not sure what backprop mechanism makes it so that the model loss collapses this spectacularly, the loss values are 
suddenly so high that they drop off the "ignore outliers" scale. The spikes dwarf the rest of the curve.

Instead, I tried a new layer layout. I keep the somewhat larger (de)conv layers, but the last layer in the conv network
is already part of the bottleneck. The before-last 128-filter (or more)  encoder layer is followed by a last conv layer
of stride 1 that has _fewer_ filters than the before-last, but a number of filters that matches the embedding size as
required by the VQ-VAE architecture. This appears to work very well: already the latest run 2022-06-24_07h22m36s that I
started dips below my best run so far on the first quarter of the training session. Whether it manages to settle into a
loss lower than my best run so far at the tail end of the run remains to be seen in a couple of hours, but so far the
run looks promising. 

UPDATE: run 2022-06-24_07h22m36s suffered from the same collapse as the previous one. I'm lowering the learning rate
from 3e-4 to 1e-4 in the hopes that that resolves the situation, otherwise I'm going to try and increase the size of the
codebook in the hope that matters will improve. I also scheduled a new run with the same configuration as 
2022-06-24_07h22m36s but with a codebook of 1024 embeddings instead of 512, just to see what it does. The scheduling
effect of Kubernetes and Argo are very, very nice. Argo keeps the workflow in a queue, as I only have one GPU machine in
my ML pool to save costs. But still it will schedule the workflow as soon as the previous one has finished. I'm starting
to benefit from my cloud setup. It cost me quite a few days of churn to get it up and running, but I really like my
cloud-MLOps setup, it's awesome.

Also, on the "good news" side is that the training time of my models _halved_ over night from the 14th to the 15th. I'm
looking into what changed during that time: I changed on both the 14th and the 15th the encoder output "rows" and 
"columns" to both half those of previously. Maybe this caused better memory allocation on the GPU or something, but 
since then, training has sped up by a factor of two, which is of course very nice.

I also traced the origin of the 'KeyError': it occurs:
- when **re-applying** the "MLOps" custom resource
- after a re-training (training resume) on a previously trained model
- followed by a training session for a fresh model, leaving the `resume_model` key blank

The trouble originates not in the configuration of the configmap, which is now bound and released uniquely for a 
specific workflow run, but in re-applying the custom resource. Kubernetes does a strategic merge on the spec of this 
resource, but fails to (re)set the `resume_model` key when it is empty (where it previously was not). I'm going to have
to look this up in StackOverflow or the kubectl GitHub issues list or something to figure out what is going on. 
Meanwhile, I have a practical workaround by removing the previous workflow before applying the new one.

Even more good news: I have a fully working and passing integration/e2e test suite for the entire pipeline. I didn't run
`coverage` but it should be at least 90% as both parts of the pipeline are subjected to a test run. 

## 2022-06-23
In an oversight I missed an important feature of the combined pipeline architecture. The encoder output channels size 
must match the latent codebook size. This is because the pixelcnn does a linear-algebraic distance operation using the
encoder outputs and the codebook, in order to 'fetch' the codebook sequences as a 'list' of pixels it autoregressively 
iterates over. In my last attempts, I had codebook 'entries' of size 64, where my encoder output had 128 channels. How
this trains at all, I'm not certain yet but at the very least it does not work on the second (pixelcnn) stage of the
pipeline.

So, I trained the same 128-channel encoder output on size-128 codebook entries, but it didn't fare so well. The images
are blurry and the vq_vae loss fails to decrease over the last, bottoming out at about epoch 96 of 128. So, back to
adding another stride-1 layer to the encoder and decoder to see if this improves matters.   

## 2022-06-22
Yesterday's run ended prematurely because of the spot instance being pre-empted, this is the first time I've actually
encountered this. The workflow process-argomlops-7xwdh was marked as failed "in response to imminent node shutdown",
which can mean a lot of things, but I presume this is because of spot instance mechanics. Due to the way I set up my
workflows, this is totally fine. The only thing I don't have now is a script archive of the exact used configuration,
but that's no big deal. The workflow ran 123 out of 128 epochs, which is fine. The results, however, are just not quite
there. There's too much blur in the images to really make sense of it. So I'm going to experiment some more with a
slightly larger network with an extra (de)conv layer. Run 2022-06-22_08h29m42s has this extra conv/deconv layer just
before the largest layer with a stride of 1, just to increase the number of layers in the network.

## 2022-06-21
From the looks of run 2022-06-20_08h02m53s and follow-up re-trained 2022-06-20_14h43m28s a quantized output of
(batch, 40, 40) on only three layers of (de)convolution for the encoder and decoder is just about enough. However I
introduced a bug in the reconstruction image callback trying to fix an earlier bug, so from the latest run I don't have
any images yet. Easiest solution is to both see if the model will learn anything useful in another 128 epochs and fix
the bug so that it produces reconstructions to properly inspect the VQ-VAE output.

Fixed the bug and inspected early re-training of 2022-06-20_14h43m28s in 2022-06-21_08h06m38s. The results are a little
on the rough side. They don't compare favourably with 2022-06-21_08h06m38s but that one used two extra (de)conv layers
in the auto-encoder architecture and took twice the time to plough through 128 epochs. So, 2022-06-21_08h06m38s is a lot
faster to train with much fewer trainable weights, which I hope should be good enough to generate artful images and 
video. I'd very much like the model to produce reconstructions in the loss-region of roughly 4e-3, which could be 
possible using a simple three-layer decoder but it's far from guaranteed. Inserting one extra layer could do the trick
as well, we'll see. 

## 2022-06-19
I tried run 2022-06-18_15h24m12s as a follow-up of 2022-06-15_09h09m47s but it didn't quite work out so well. The model 
has too much trouble making sense of (batch, 20, 20) sized quantized output, there's just not enough expressive power in
output this size I guess. Instead, I'll try to tweak a (batch, 40, 40) model and see how far I can compress it.

## 2022-06-16
2022-06-15_09h09m47s (reconstruction loss: 8.4e-3 - vq_vae_loss: 2.9e-2) was the first attempt at producing a 
(batch, 20, 20) sized quantized output. The reconstruction loss is again worse than the (batch, 40, 40) sized one, but at
least the loss in reconstruction accuracy seems consistent and with 128 epochs of 256 batches of size 16 samples each,
the model does show room for improvement on the tail end of the run. So, time to experiment some more and resume training
on some models, I think (batch, 20, 20) is fine for an output size, even the mnist example of the VQVAE had (batch, 7, 7)
which is rather large considering the inputs of (batch, 28, 28, 1). I also see that I have part of my network as stride
1 layers but with more filters, so I think I'm going to try to reduce the number of parameters in the network a bit by
cutting them out and seeing how the model fares. I might have to do a re-training though of the run mentioned above to
make sure that any follow-up attempts are able to produce reconstructions that are still of artistic value. The 
reconstructions now have a little bit too much vagueness to be still of use.

## 2022-06-15
Run 2022-06-14_10h16m16s is the first try to lower the dimensionality of the encoder output. The output was compressed
by adding an extra convolution layer in both encoder and decoder with stride 2, halving the quantized output size from
(batch, 80, 80) (2022-06-13_09h01m09s) to (batch, 40, 40), meaning that pixelcnn sampling will speed up fourfold. The
reconstruction loss wasn't quite that of the 2022-06-13_09h01m09s run (3.3e-3 vs 5.5e-3 for 2022-06-13_09h01m09s) but it
can't be ruled out that this was just a fluke. The "dark image spike" that always seems to occur during a 
training-from-scratch session was rather late and the model was still recovering. The results, althoug visibly worse 
than the "high quality" run 2022-06-13_09h01m09s are still "artistically sound" in the sense that I rather like the
slight bluriness. I'm going to add another convolution layer to see if I can compress to encoder output size 
(batch, 20, 20) and see if the results are still acceptable.

Also, I started to fix a bug in my "argo operator" setup. I noticed that sometimes the new values in the configmap with
the manifest MLOps settings weren't set properly. This only happens if the previous run re-trained a model using the
`resume_model: gs://bucket/model/path` setting, and the next run trained from scratch, using `resume_model: null` or
just an empty `resume_model:`. In those cases, the `resume_model` key is mysteriously dropped from the configmap, I 
can't figure why this would be the case but that's Kubernetes for you, expect weird YAML errors and strange bugs from
time to time. Instead of applying the new configuration to the existing configmap, the workflow starts by deleting the
old configmap and then applying the new settings from scratch. This appears to resolve the issue. Another solution could
be to delete the configmap after the workflow has run, but this hampers inspection and transparency of faild workflow
runs. Although the contents of the manifest are also logged in the Argo UI and logs. Maybe it's not necessary to keep
the configmap around after the workflow has run.

Best solution would be to create a per-run configmap and pass the name of the configmap to the workflow template call
step, this would allow multiple MLOps workflows to run at the same time without incurring data races on a shared
configmap.

## 2022-06-14
Run 2022-06-13_09h01m09s used 512 embeddings of size 64 instead of the previous 256 embeddings of size 128. It had little
trouble making sense of the data, it trained a good looking curve for almost 12 hours to a reconstruction loss of 3.3e-3
with a vq-vae loss of 4.2e-3. It still shows a nicely downward curve at the end so it isn't done training either, I'll
give it another session of that size to see if the reconstructions improve further. This should set me up nicely for
training a beter pixelcnn, one that has to deal with embeddings of only size 64 instead of the previous 256. 

As for visual inspection of the results, they look almost photo-realistic. Now I may not even re-train the model again,
I don't think I would like it to be even more realistic than this. It certainly looks like the encoder and decoder are
powerful enough that they settle for size-64 embeddings, while the number of 512 embeddings definitely helped in
diversifying the output. Let's see if we can get the pixelCNN part of the pipeline working properly.

However, I think I'm going to try another run with an extra conv layer to lower the dimensionality of the encoder 
outputs. The encoder outputs is the reconstruction target of the pixelcnn. At the moment, with my 640x640 images, it
outputs 80x80 encodings, but feeding these to the pixelcnn is too much time-consuming since it is autoregressive. 
Halving this should help in making epoch callback reconstructions and generating new examples as well.

## 2022-06-13
Implemented the pixelcnn part so that it works with my config.yaml hyperparameters and deployment config file. I ran a
short tryout with the model and I see now that I got a few things backwards. I'm using "only" 256 embeddings with a size
each of 128 each. The number of embeddings should probably be larger. The input space is probably a lot smaller than can
be expressed with size 128, that's about half a full DALL-E configuration with an enormous range of image subjects. FAFA
is far more constrained than generic DALL-E likes. So, it should work with 128 sized, 92-sized or even 64-sized 
embeddings. However, the details in the images could benefit from a larger number of embeddings, say 256. So, I'm going 
to redo the vq-vae part with lower-dimensional but more embeddings.


## 2022-06-07
Re-training models works, but now I have to keep track of some provenance. I trained session 2022-06-03_09h39m59s on
2022-06-02_16h44m41s (which comes from 2022-04-21_10h10m14s) and it improved a little on the reconstruction loss, from
~3.12e-3 to 2.77e-3. So this is a 0.35e-3 improvement, or 11% improvement, over the course of a 5.5 hrs and 64 epochs of
256 batches each. I think I'm going to stop here for now, there's enough detail in the reconstructions to clearly
discern human figures but not enough to make realistic faces. I'm still clear of 
['uncanny valley'](https://en.wikipedia.org/wiki/Uncanny_valley) I hope. The figures aren't ghostly in any sense anymore
and the reconstruction loss is going down, while the vq-vae-loss is steadily increasing now. It might signify some sense
of overfitting (?) but I think it's alright to stop here and move to the pixelCNN part of the pipeline, I can always go
back and do some more training but I'll have to re-do the entire pixelCNN part if I do.

## 2022-06-03
It worked! I re-implemented my ML project in Argo and it worked out very nicely. The workflow uses a spot image to save
consts and it saved checkpoints directly to my artifacts bucket.

I also implemented some other useful stuff: re-training an existing model checkpoint. Loading it from the bucket goes
perfectly fine, and the checkpoints are also saved in the correct location. Re-training did not result in a significant
drop in overall loss, but at least now I know that the model is about as well-trained under the current hyperparameter
settings as can be. I think I'm going to do some tuning before starting the pixelCNN part of the project. The part of 
the model that _did_ show significant improvement, was the reconstruction loss, which is great. It dropped from 5.2e-3
to 3.08e-3, so it nearly halved. That was well worth investigating. I'm going to see if I can get it down a little bit
more using a lower learning rate.

## 2022-06-02
Oh yeah I had this journal. Been busy doing things that actually pay the bills. A good project, but whatever. I spent
quite a few days implementing the Argo part for this endeavour. The main takeaway is that it is very complex, highly
time-consuming and very satisfying. Now my ML setup includes its own operator that I can re-use for other projects. I
may consider moving it into its own repository. There are still a few things to iron out, but the basic mechanism is
very good. I'm far enough that I can
- Execute arbitrary code from public git repos. This allows me to swap one project for another easily
- All kubernetes stuff is separated from the script part. The stuff in the config manifest intended for the script is
  mostly in `models` and `data`, since the script should only/mostly be involved in handling models and data.
- All k8s stuff is in the rest of the manifest. The metadata, the image to use, etc.
- All infrastructure requirements are handled by the operator. It's a simple enough Helm chart that manages configmaps,
  storage, machine access (GPU resources etc). Argo handles the administration: event handling, logging and archiving of
  artifacts not handled by the ML script
So, now I have a proper MLOps setup that I can re-use for basically any ML project I undertake, which is pretty nice.

## 2022-05-10
I suspended work on FAFA-VAE of a moment to take some days off and to work on things that actually get the bills paid.
Also, I started investigating some more into the VQ-VAE architecture and on why it is a two-stage process. There was an
existing stackoverflow question on this: 

## 2022-04-21
From the increased learning rate of 2e-4 I can conclude that it did not really help. The initial curve angle is steeper,
but the learning process appears to be more erratic. Intermittent spikes in loss increase (once from 0.015 to 0.8) in
the reconstruction loss occur, after which the model tries to settle back to loss values that match the slightly lower
learning rate of 1e-4. I'm not sure what the model does here, but when I inspect the images at this point, the model
clearly suffered some kind of collapse because it appears as though all it has learned is lost: it produces stick
figures again.

Looks as though I'm burning through my starting freebie credits as well. I'm losing about € 10 a day on a single
experiment of about 12 hours of training, and that's the minimum I can do to get a somewhat clear picture of what the
model learning accomplishes. This means I'll be out of budget by the end of this month and I'll either have to switch to
training on my good old mobile GTX 1060 or decide to spend some of my own budget on a T4 again. I'll certainly want to 
keep the storage bucket I think, it's a very useful way of storing the training artifacts. Training sessions now cost
about €0,40 for a T4 on an n2-4 machine, with sessions lasting for ~12hrs is about €5 a day without vm disk and bucket
usage. It's certainly worth looking into pre-emptible machines, but I want to set up a machine learning Kubernetes
environment for that first, I think, to automatically reschedule failed training jobs.

Today I'm going to drop the learning rate back to 1e-4 and also I'm going to drop the 'standing' filter on the images. I
want to see how the model does on the much more complete image collection. I think it should do fine, and if it does I
can start re-training sessions instead of learning from scratch every time. Started run 
gs://antfield/FAFA/artifacts/2022-04-22_09h33m57s/ in order to follow this plan.

Also changed the call to `model.fit`, to set the number of workers to 8 and the max queue size to twice that of a batch
size. Maybe this will speed up the learning process and get rid of the 'on batch end is slow compared to batch training'
kind messages.

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
After try 1 the VQ-VAE already shows much, much better reconstructions than the standard VAE. The 256 epochs I train for,
show much clearer figures. Interestingly, the model focuses much more on the general picture than on trivial things,
such as logos and watermarks. It shows decent quality images and it can certainly improve with a bit more training, for
sure. This is all on a 4-layer conv mirrored to deconv on the decoder, with the largest layer having 64 filters. It'd
definitely be worth adding another 64 filter, or even a 128 filter layer.

Thing is, though, that I'm still pretty clueless on the role of the pixelCNN. Until I know what it does, I'm not sure if
it's worth training deeper VAE nets. 

## 2022-04-14
I've successfully refactored the autoencoder model to a vector-quantized VAE but it took me about a day to structure it
to my implementation. The changes are very, very complex, and that is saying something when starting out from a VAE to
begin with. The concept of the VQ-VAE 'codebook' in itself is very complex, although it is maybe best described as an
one-hot embedding space (hence the 'quantized' predicate) that encodes the input feature space in a lower-dimensional
latent space. 

With the new implementation, of course now my previous loss figures are utterly meaningless and I have to inspect the
reconstructed images to see how it fares. One interesting thing is that the number of weights on the VQ-VAE is much,
much lower than on the vanilla VAE. The VAE uses a flattening operation and a subsequent fully connected layer close to
the latent, as the latent is a simple vector. The fully connected layer, however, was absolutely humongous, given the
size of the images on the one hand, and the size of the latent vector on the other. At least that is solved in the
VQ-VAE implementation, but I may have to add in some conv/deconv layers to give the model enough weights to learn a good
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

