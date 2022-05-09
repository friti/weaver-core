# Prodouce a docker image with weaver

* First step is install weaver locally following the recipe given in the main README file at [Weaver Installation](https://github.com/rgerosa/weaver/blob/dev_sim_class_reg/docker/README.md)
* Once installed Weaver within a conda environment, dependent packages can be exported via:
  ```sh
  conda activate weaver
  conda env export > weaver-environment.yaml
  conda deactivate
  ```
*The Docker image is created by describing the installation procedure in the Docker file. A .dockerignore file can be used express caches/packages not to be included in the docker container.
* By default the docker images and corresponding containers are created on the local machine file system in /var/lib/docker/. Therefore, you need to create the container on a machine for which you have root priviledges and enough space to host a container with a total size of about 8-15 Gb depending on how many conda/pip packages you want to include.
* The Docker file is contained in this directory. To create the container and then execute it locally:
  ```sh
  # Build container
  sudo docker build -t <name of the container> .
  # Display images and containes
  sudo docker images
  sudo docker containers
  # Run container locally
  sudo docker run -i -t <name of the container> /bin/bash
  ```
* It is commonly good practise to store Docker images in github/gitlab repositories if your account has enough space. In this case you can upload to gitlab and then reclaim space on the local machine as follows:
  ```sh
  sudo docker login <git repository>
  sudo docker push <git repository/account name/name of the container>
  sudo docker system prune --all --force   
  ```