## Installation

The Umami framework can be installed either locally or be run by using a Docker image.

Below, both options are outlined.

### Local installation

First, retrieve the project by cloning the git repository. Then, install the project locally.

```bash
git clone --recursive ssh://git@gitlab.cern.ch:7999/atlas-flavor-tagging-tools/algorithms/umami.git

python setup.py install

```

Alternatively, if you want to develop the code, use the `develop` install command, which creates a symbolic link to the local directory instead of copying it.
Consequently, any changes you make to the code are directly picked up.

```bash
python setup.py develop
```

On certain clusters `Singularity` might be configured such that it is not writable and `python setup.py develop` will fail. In this case you need to set your `PYTHONPATH` to e.g. the current directory (`export PYTHONPATH=$PWD:$PYTHONPATH`) and choose the current folder also as install directory via `python setup.py develop --install-dir .`. It can then also happen that you are getting a weird error with `RecursionError: maximum recursion depth exceeded in comparison`, then you need to clean up your repository via ` rm -rf umami.egg-*`.

### Docker container
You can run Umami in a [Docker container](https://www.docker.com/resources/what-container). This is the most convenient way and ensures that you are not required to install any dependencies as those are already included in the Docker image.

The images are created automatically from the `master` branch and updated for every modification using Continuous Integration. Here, the `latest` tag on Docker Hub corresponds to the `master` branch in the GitLab project. Similarly, the `latest-gpu` tag on Docker Hub corresponds to the `master` branch but provides additional support for running TensorFlow with GPUs.
Other tags correspond to the tags in the GitLab project.

#### Launching containers using Docker (local machine)
If you work on a local machine with Docker installed, you can run Umami with this command:
```bash
docker run --rm -it btagging/umami:latest
```

You can mount local directories with the `-v` argument:
```bash
docker run --rm -it -v /cvmfs:/cvmfs -v /afs:/afs -v $PWD:/home/workdir btagging/umami:latest
```

There is also an image with GPU support, which can significantly speed up the training step assuming your machine has a GPU.
You can run Umami image with GPU support using this command:

```bash
docker run --rm -it btagging/umami:latest-gpu
```

#### Launching containers using Singularity (lxplus/institute cluster)
If you work on a node of your institute's computing centre or on CERN's `lxplus`, you don't have access to Docker.
Instead, you can use [singularity](https://sylabs.io/guides/3.7/user-guide/introduction.html), which provides similar features.

You can run Umami in singularity with the following command:
```bash
singularity exec docker://btagging/umami:latest bash
```

Alternatively, you can retrieve the image from the GitLab container registry
```bash
singularity exec docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:latest bash
```

#### Launching containers with GPU support using Singularity (lxplus/institute cluster)

The image with GPU support can be run with the command (note that singularity requires the `--nv` argument to provide the GPU resources within the container):
```bash
singularity exec --nv docker://btagging/umami:latest-gpu bash
```

If you work on AFS and want to explicitly state which paths should be made available, consider the `--contain` argument and mounting volumes inside the container with the `--bind` argument:
```bash
singularity exec --contain --bind /afs  --bind /cvmfs --bind /eos docker://btagging/umami:latest bash
```
