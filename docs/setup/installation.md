It is recommended to use the Umami framework with a Docker image.

You can either choose to use a fully build version where you are ready to run the code or a version where you are able to modify the code.

If you are using lxplus or a local cluster with `cvmfs` access, the recommended way is to use the images unpacked on `cvmfs` as described [here](#singularity-images-on-cvmfs). This will just work out of the box.


## Docker container
You can run Umami in a [Docker container](https://www.docker.com/resources/what-container). This is the most convenient way and ensures that you are not required to install any dependencies as those are already included in the Docker image.

The images are created automatically from the `master` branch and updated for every modification using Continuous Integration. Here, the `latest` tag on [Docker Hub](https://hub.docker.com/r/btagging/umami) corresponds to the `master` branch in the GitLab project. Similarly, the `latest-gpu` tag on Docker Hub corresponds to the `master` branch but provides additional support for running TensorFlow with GPUs.
Other tags correspond to the tags in the GitLab project. For more details see the image overviews below.


There are three different kind of images:
- Base images
    - these image types contain all the necessary dependencies for `umami` but not the `umami` package itself
    - these are best suited for any developemts in `umami`
    - You can browse them [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/container_registry/8906) in the gitlab container registry
- Base images plus
    - these image types extend the `umamibase` image installing additional packages defined in [requirements_additional.txt](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/requirements_additional.txt)
    - these are especially useful for R&D studies requiring additional packages
- Packaged images
    - these images use the base images and have `umami` installed on top
    - these are the best choice if you just want to run `umami` but you don't want to change anything in the code
    - You can browse them [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/container_registry/8387) in the gitlab container registry


??? info "Overview Base images"
    | Image tag | Description  | gitlab registry |
    | ---------- | ----------  | ---------- |
    | `latest` | CPU base image from `master` |  gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase:latest|
    | `latest-gpu` | GPU base image from `master` |  gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase:latest-gpu|
    | `latest-pytorch-gpu` | GPU base image from `master` with pytorch instead of tensorflow |  gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase:latest-pytorch-gpu|
    | `0-2` | CPU base image of tag `0.2` |  gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase:0-2|
    | `0-2-gpu` | GPU base image of tag `0.2` |  gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase:0-2-gpu|
    | `0-1` | CPU base image of tag `0.1` |  gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase:0-1|
    | `0-1-gpu` | GPU base image of tag `0.1` |  gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase:0-1-gpu|

    *not all tags are listed here, please have a look in the [gitlab registry](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/container_registry) for a complete list*


??? info "Overview BasePlus images"
    | Image tag | Description  | gitlab registry |
    | ---------- | ----------  | ---------- |
    | `latest` | CPU base image from `master` |  gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase-plus:latest|
    | `latest-gpu` | GPU base image from `master` |  gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase-plus:latest-gpu|

    *not all tags are listed here, please have a look in the [gitlab registry](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/container_registry) for a complete list*


??? info "Overview Packaged images"
    | Image tag | Description | Docker hub | gitlab registry |
    | ---------- | ----------  | ---------- |  ---------- |
    | `latest` | CPU packaged image from `master` | btagging/umami:latest | gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:latest|
    | `latest-gpu` | GPU packaged image from `master` | btagging/umami:latest-gpu | gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:latest-gpu|
    | `latest-pytorch-gpu` | GPU packaged image from `master` with pytorch instead of tensorflow | btagging/umami:latest-pytorch-gpu | gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:latest-pytorch-gpu|
    | `0-2` | CPU packaged image of tag `0.2` | btagging/umami:0-2 | gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:0-2|
    | `0-2-gpu` | GPU packaged image of tag `0.2` | btagging/umami:0-2-gpu | gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:0-2-gpu|
    | `0-1` | CPU packaged image of tag `0.1` | btagging/umami:0-1 | gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:0-1|
    | `0-1-gpu` | GPU packaged image of tag `0.1` | btagging/umami:0-1-gpu | gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:0-1-gpu|
    | `jupyter-develop` | CPU packaged image of tag `jupyter-develop` | -- | gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:jupyter-develop|

    *not all tags are listed here, please have a look in the [gitlab registry](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/container_registry) for a complete list*




### Launching containers using Docker (local machine)
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

???+ info "Using the base image instead"
    As mentioned before, if you want to modify the code, please use the Base images which would change your docker command e.g. to
    ```bash
    docker run --rm -it gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase:latest
    ```

### Launching containers using Singularity (lxplus/institute cluster)
If you work on a node of your institute's computing centre or on CERN's `lxplus`, you don't have access to Docker.
Instead, you can use [singularity](https://sylabs.io/guides/3.7/user-guide/introduction.html), which provides similar features. How to use singularity on `lxplus` can be found [here](https://hsf-training.github.io/hsf-training-docker/10-singularity/index.html)

You can run Umami in singularity with the following command:
```bash
singularity exec docker://btagging/umami:latest bash
```

Alternatively, you can retrieve the image from the GitLab container registry
```bash
singularity exec docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:latest bash
```

???+ info "Using the base image instead"
    ```bash
    singularity exec docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase:latest
    ```

???+ info "Speeding up loading time of singularity images"
    The above commands are often slow since they require a new conversion of the docker image to a singularity image.
    There is a possibility to avoid this by converting them once via
    ```bash
    singularity pull <folder_where_you_want_to_store_the_image>/umami_base_cpu.img docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase:latest
    ```
    Afterwards you can use this local image e.g. like that
    ```bash
    singularity exec <folder_where_you_want_to_store_the_image>/umami_base_cpu.img bash
    ```

???+ warning "Singularity cache and temporary files"
    By default, `singularity` will store the cache in your home directory, which will cause space issues.
    You can redirect the persistent cache to another folder via `export SINGULARITY_CACHEDIR=<alternative_folder>` which could be for instance the EOS/CERNBox area, instead of home on `lxplus`. You will still want to keep the 'temporary' location on (fast) local storage, this is controlled by `SINGULARITY_TMPDIR`.
    The easiest is to add to your `~/.bashrc` on `lxplus` the following lines
    ```bash
    export SINGULARITY_CACHEDIR=/eos/user/${USER:0:1}/${USER}/singularity
    export SINGULARITY_TMPDIR=/tmp/${USER}/singularity
    mkdir -p "${SINGULARITY_TMPDIR}"
    ```
    after having created the (persistent) cache folder via
    ```
    mkdir /eos/user/${USER:0:1}/${USER}/singularity
    ```
    In case you don't yet have `/eos/user` space, you can do that by connecting to https://cernbox.cern.ch


### Singularity images on CVMFS

It is also possible to use the `umami` and `umamibase-plus` images directly from `cvmfs`. If you have a good connection to `cvmfs` this will be very fast to use out of the box.

The `umamibase-plus` images are located in the folder `/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/`. You can use the latest image e.g. via

```bash
singularity shell /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase-plus:latest
```

??? info "Available tags"
    To check which tags are available, you can have a look in the folder via
    ```bash
    ls /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/
    ```


The `umami` images are located in `/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/`.
An example usage of the `latest` image would be

```bash
singularity shell /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:latest
```

??? info "Available tags"
    To check which tags are available, you can have a look in the folder via
    ```bash
    ls /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/
    ```




### Launching containers with GPU support using Singularity (lxplus/institute cluster)

The image with GPU support can be run with the command (note that singularity requires the `--nv` argument to provide the GPU resources within the container):
```bash
singularity exec --nv docker://btagging/umami:latest-gpu bash
```

If you work on AFS and want to explicitly state which paths should be made available, consider the `--contain` argument and mounting volumes inside the container with the `--bind` argument:
```bash
singularity exec --contain --nv --bind /afs  --bind /cvmfs --bind /eos docker://btagging/umami:latest bash
```

???+ info "Using the base image instead"
    ```bash
    singularity exec --nv docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase:latest-gpu bash
    ```


## Cloning the repository
As mentioned before, if you plan to modify the code, the best choice are the Base images. How to use images on your local machine or cluster you can see below.

In order to make code changes, you need to clone the repository via
```bash
git clone ssh://git@gitlab.cern.ch:7999/atlas-flavor-tagging-tools/algorithms/umami.git
```
to make the `umami` package now accessible in python you need to run
```bash
python -m pip install -e .
```
within the `umami` folder.


???+ info "Newer singularity versions"
    On certain clusters `Singularity` might be configured such that it is not writable and `python -m pip install -e .` will fail. In this case you need to set your `PYTHONPATH` to e.g. the current directory (`export PYTHONPATH=$PWD:$PYTHONPATH`) and choose a folder e.g. `python_install` also as install directory via `python -m pip install --prefix python_install  -e .` (you first need to create the `python_install` folder). It can then also happen that you are getting a weird error with `RecursionError: maximum recursion depth exceeded in comparison`, then you need to clean up your repository via ` rm -rf umami.egg-*`.

    This is also bundled in a script you can use
    ```bash
    source run_setup.py
    ```




??? info "Local installation"
    This option is very much discourraged due to the difficulty of installing tensorflow properly which is handled automatically by the docker containers.


    First, retrieve the project by cloning the git repository.

    ```bash
    git clone ssh://git@gitlab.cern.ch:7999/atlas-flavor-tagging-tools/algorithms/umami.git
    ```

    In order for the `umami` code to work you need at least `python3.8` or higher.

    Now you need to install all the requirements which are in the first place `tensorflow`.

    WARNING: it is typically quite complicated to install `tensorflow` properly, especially on a cluster. You might consider using a conda environment for this.

    Tensorflow, together with all other requirements can be installed using the provided `requirements.txt` file and the following command:
    ```
    pip install -r requirements.txt
    ```

    Then, install the project locally.
    ```bash
    python -m pip install .
    ```
    Alternatively, if you want to develop the code, use the `-e` option, which creates a symbolic link to the local directory instead of copying it.
    Consequently, any changes you make to the code are directly picked up.

    ```bash
    python -m pip install -e .
    ```

    The `requirements.txt` serves as a `lock` file, snap-shoting the actual packages installed with their versions. Together with the related hashes, the environment can be exactly rebuild. The process to generate this file is explained in the [development](https://umami-docs.web.cern.ch/setup/development/#updating-ci-files) part of the documentation.
