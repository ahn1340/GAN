# GAN
Pytorch implementation of the [DCGAN paper][1] and [Improved Techniques for Training GANs][2].
Currently, only feature matching is implemented. Minibatch discrimination, label smoothing and historical averaging
will be implemented soon.

[1]: https://arxiv.org/abs/1511.06434  "DCGAN"
[2]: https://arxiv.org/abs/1606.03498 "Improved Techniques"

# Demo
### CelebA dataset (vanilla adversarial loss)
![Alt Text](./demo/progress_adversarial_100ep.gif)

### CelebA dataset (feature matching loss)
![Alt Text](./demo/progress_feature_matching_100ep.gif)

# How to use
Put your images in a folder and specify path to that folder in config/hyperparameters.yaml.
Also, other training configurations can be modified in the file.
Then run python main.py.


Training progress will be logged in WandB.

# TODO
* Minibatch discrimination
* Label smoothing
* Historical averaging

