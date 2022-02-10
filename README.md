# H.IAAC-M4-Manifold-Learning-Tools

This repository contains all tools developed for H.IAAC-M4

Check some tutorials at tutorials directory!

## Useful Material to Check if You are Not to Familiar...

1. Numpy and Matplotlib tutorial: https://cs231n.github.io/python-numpy-tutorial/

2. Pandas tutorial: https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html

3. Scikit-Learn ....

4. Some useful Linux commands: https://maker.pro/linux/tutorial/basic-linux-commands-for-beginners

5. SSH: https://21-lessons.com/the-ssh-beginners-primer/

6. GIT: https://www.simplilearn.com/tutorials/git-tutorial/git-tutorial-for-beginner?source=sl_frs_nav_playlist_video_clicked, https://www.nobledesktop.com/learn/git/git-branches

7. Docker: https://www.simplilearn.com/tutorials/docker-tutorial/getting-started-with-docker?source=sl_frs_nav_playlist_video_clicked



## Install Instructions

In order to use the available tools, you need to build the docker image and run it. The available Dockerfile and scripts create a Docker image with all required packages and libraries and start a Jupyter lab server, listening at 8888.

### Cloning the GIT

First you must clone the git to the machine where you will work on. To do this, you can run the following command:

```
git clone https://github.com/lmcad-unicamp/H.IAAC-M4-Manifold-Learning-Tools.git
```

The directory `H.IAAC-M4-Manifold-Learning-Tools` will be create with all the content. You must go to the directory to perform all operations, using

```
cd H.IAAC-M4-Manifold-Learning-Tools
```

Note that all used datasets are inside `data` directory and, hence, the size of the repository will be large (a few gigabytes).

#### Using Different Branches

Multiple users can work on different branches of the code. You can list all branches (on your local computer and on remote server) using the command below:

```
git branch -a
```

You can switch to another branch using the `checkout` command. For instance, to change to `borin` branch, you can use the following command:

```
git checkout borin
```

Doing changes on a branch does not affect another one, as branches are independent.

Besides that, changes are only updated to the server after the `git push` command. To perform a push, you must first do a `git commit` on files that you have changed in your branch. Commit will log all the alterations in your files. In this way, data is never lost and you can travel back to a specific point as you want. So, always try to do a lot of small commits than larger ones.

### Setting Up Docker

A docker image is available to use all tools properly. The image comes with all packages and libraries installed, required to run the experiments. To build the docker image, a build script is made available. You can create the image called `hiaac4-full`, using the command below:

```
./build_docker.sh
```

You can check the docker images using the `docker images` command. Usually, a image must be build only once and you can instantiate as many containers as you want based on this image.

After the image is created, you must create a container using that image and start the jupyter-lab server. To do this, a script was made available and you can just run the command below:

```
./run_docker.sh
```

This script will:
* Start the jupyter-lab server, listing on port 8888. You can access a browser (e.g. chrome) on the machine where the container is running and access the address `localhost:8888` to access the jupyter-lab. If you are running the docker container on another machine, you must do a Port Forwarding step (instructions in later sections)
* Map the current directory to the container. In this way, every change made in the container will be reflected in the directory. For instance, if you create a file inside the container, the file will be created in the `H.IAAC-M4-Manifold-Learning-Tools` directory.

You can check the running containers using the `docker ps` command.



## Starting Jupyter-lab Server on Remote Machines

To execute jupyter-lab in a remote machine, first you must connect to the that machine via SSH, preferably, without asking password using your identity file (SSH public key). The following steps will show:

1. How to create an SSH key file and put it in a remote machine
2. How to create an SSH config file with the hosts
3. How to start the jupyter-lab server in the remote machine and access the jupyer-lab in the browser of your local machine.

For simplicity we will refer to:
* Remote Machine (RMACHINE) &rarr; Remote machine where you want to login (and start the jupyter-lab). It could be an IP (e.g. 200.40.30.10) or an host (e.g. machine.ic.lmcad.unicamp.br)
* Remote user (RUSER) &rarr; Name of the user used to login into the remote machine
* Remote password (RPASS) &rarr; Password used to login into the remote machine

### Creating an SSH Key and Putting it in a Remote Machine (optional)


The simplest method to test your login into a remote machine is using the following command to login as a RUSER, using your RPASS:

```
ssh RUSER@RMACHINE
```

Typically, some system administrators does not allow to login with this method (using RUSER and RPASS) and you must login based on your identity, via SSH. To do this, usually you will have two keys (a keypair) in your local machine:

* A public key, that is your public identity, and;
* A private key, that verify that the holder of this public key is, definitively, you.

For default configurations, your keypair usually resides on `~/.ssh/` directory (`~` stands for the user's home directory), where the `~/.ssh/id_rsa` is your private key and the `~/.ssh/id_rsa.pub` is your public key. If none of these files exists you can simply generate them using the following command, and just pressing enter to all questions (leaving all blank).

```
ssh-keygen
```

Your keypair is unique and uniquely identify you. Be sure to do not lose it!

After the keypair is generated, you must copy your public key to the RMACHINE and tell the RMACHINE to authorize the user who possess the private key (corresponding to that public key) to login in the RMACHINE. To do this, you can use the following command:

```
ssh-copy-id -i ~/.ssh/id_rsa.pub RUSER@RMACHINE
```

Note that, usually in a machine shared with several users, where an administrator manages the access, you must inform to him your public key he will put your public key in the RMACHINE.

Once your public key is authorized in the RMACHINE, you can login to it using the command:

```
ssh RUSER@RMACHINE
```

In this step, no password should be asked.

### Creating an SSH config File (optional)

You can login faster by creating an ssh config file. To do this, with the default SSH configuration, create a file at `~/.ssh/config`. At this file you can add your hosts RMACHINEs to login faster.
Open the `~/.ssh/config` and put one host configuration, similar to the one below:

```
Host XXX
	Hostname RMACHINE
	User RUSER
	Port YYY
	IdentityFile ~/.ssh/id_rsa
```

Where the XXX is a name for the RMACHINE (you decide it) and the "Port" line is optional. Doing this you can login into the RMACHINE simply running:

```
ssh XXX
```

### Starting Jupyter-lab in a Remote Machine and Accessing it in your Local Machine

1. Login into the RMACHINE and follow the install instructions above (install docker, clone the git, etc.) to build the docker image.
2. Start the jupyter-lab server into the RMACHINE, using the `run_docker` command script (as in the install instructions). You can check if you container is running using the `docker ps` command.
3. In your local machine you must perform an [SSH Port Forwarding](https://www.ssh.com/academy/ssh/tunneling/example). SSH port forwarding is a mechanism in SSH for tunneling application ports from the local machine to the remote machine, or vice versa. In our case, if jupyter-lab server start at RMACHINE listening at port 8888, we want that, every connection in our local machine in port 8888 be redirected to the RMACHINE's port 8888. To do this you can use the following command:

 ```
 ssh -N -T -L -v 8888:localhost:8888 XXX
 ```

Where `XXX` is the XXX is a name for the RMACHINE, as in the `~/.ssh/config` file, or the `RUSER@RMACHINE`. Note that, the port forwarding will execute until you close your terminal.


## Tips

### Keep jupyter-lab alive

You may want to keep jupyterlab running indefinetly on the server machine. You can use tmux and [dettach a session](https://danielmiessler.com/study/tmux/).

### Using Bourne-again Shell in jupyter-lab

You may wish to use bourne again shell instead of `sh` in your jupyter lab. To do this, export `SHELL` variable before executing jupyter, using the following command.

`export SHELL=/bin/bash`
