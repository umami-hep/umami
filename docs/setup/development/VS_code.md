# Using Visual Studio Code
The editor Visual Studio Code (VSCode) provides very nice and helpful options for developing Umami. VSCode is also able to run a singularity image
with Umami and therefore has all the needed dependencies (Python interpreter, packages, etc.) at hand. A short explanation how to set this up
will be given here.

## Using a Singularity Image on a Remote Machine with Visual Studio Code
To use a Singularity image on a remote machine in VSCode, to use the Python interpreter etc., we need to set up some configs and get some
VSCode extensions. The extensions needed are:

| Extension | Mandatory | Explanation |
|-----------|-----------|-------------|
| [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) | Yes | The Remote - SSH extension lets you use any remote machine with a SSH server as your development environment. |
| [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) | Yes | The Remote - Containers extension lets you use a singularity container as a full-featured development environment. |
| [Remote - WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) | Yes (On Windows) | The Remote - WSL extension lets you use VS Code on Windows to build Linux applications that run on the Windows Subsystem for Linux (WSL). |
| [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) | Yes | The Remote Development extension pack allows you to open any folder in a container, on a remote machine, or in the Windows Subsystem for Linux (WSL) and take advantage of VS Code's full feature set. |

Now, to make everything working, you need to prepare two files. First is your ssh config (can be found in ~/.ssh/config). This file
needs to have the permission of only you are able to write/read it (`chmod 600`). In there, it can look like this for example:

```bash
Host upimage~*
    RemoteCommand singularity shell <some>/<image>/
    RequestTTY yes

Host tf2~*
    RemoteCommand source /etc/profile && module load tools/singularity/3.8 && singularity shell --nv --contain -B /work -B /home -B /tmp docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase-plus:latest-gpu
    RequestTTY yes

Host login_node
    HostName <Login_Hostname>
    User <Login_Username>
    IdentityFile <path>/<to>/<private>/<key>

Host working_node tf2~working_node upimage~working_node
    HostName <working_node_hostname>
    User <Username>
    ProxyJump login_node
```

The first and second Host here are the entries for the images you want to use. You need to define a name for the images, e.g. `tf2` and then add `~*` at the end. Two commands need to be given here. First is the `RemoteCommand`. Here you can define all the commands that are executed when logging into your working node. If you need to source a profile first or load singularity, you can simply chain the commands with `&&`. The last command of `RemoteCommand` must be the `singularity shell` command.   
The second command is the `RequestTTY`. This must be `yes`.

The third Host is the login node (if you need to login into this one before you can log into the working node. If you don't need that, you can ignore this host). Here you need to define all the things for the connection, like `HostName`, `User` or the `IdentityFile`.

The last Host is the working node. Same as for the login node, you need to add here the options needed for the connection to the working node (if you don't use the login node, remove the last line with `ProxyJump`). The name here is a bit different. You need to add multiple names here. One for each image. Just add the name of the image and the base hostname as one e.g. `tf2~working_node`.

After adapting the config file, you need to tell VSCode where to find it. This can be set in the `settings.json` of VSCode. You can find/open it in
VSCode when pressing `Ctrl + Shift + P` and start typing `settings`. You will find the option `Preferences: Open Settings (JSON)`. When selecting this,
the config json file of VSCode is opened. There you need to add the following line with the path of your ssh config file added (if the config is in the default path `~/.ssh/config`, you don't need to add this).

```json
"remote.SSH.configFile": "<path>/<to>/<ssh_config>",
"remote.SSH.remoteServerListenOnSocket": false,
"remote.SSH.enableRemoteCommand": true,
```

The second option added here disables the `ListenOnSocket` function which blocks the running of the singularity images in some cases. The third option will enable the remote command needed for singularity which is blocked when `ListenOnSocket` is `True`.

Now restart VSCode and open the Remote Explorer tab. At the top switch to `SSH Targets` and right-click on the `tf2~` connection and click on `Connect to Host in Current Window`. VSCode will now install a VSCode server on your ssh target to run on and will ask you to install your extensions on the ssh target. This will improve the performance of VSCode. It will also ask you which path to open. After that, you can open a python file and the Python extension will start and should show you at the bottom of VSCode the current Python Interpreter which is used.
If you now click on the errors and warnings right to it, the console will open where you can switch between Problems, Output, Debug Console, Terminal and Ports. In terminal should be a fresh terminal with the singularity image running. If not, check out output and switch on the right from Tasks to Remote - SSH to see the output of the ssh connection.


## Useful Extensions
| Extension | Mandatory | Explanation |
|-----------|-----------|-------------|
| [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) | Yes | A Visual Studio Code extension with rich support for the Python language (for all actively supported versions of the language: >=3.6), including features such as IntelliSense (Pylance), linting, debugging, code navigation, code formatting, refactoring, variable explorer, test explorer, and more! |
| [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) | Yes (Will be installed with Python extension) | Pylance is an extension that works alongside Python in Visual Studio Code to provide performant language support. Under the hood, Pylance is powered by Pyright, Microsoft's static type checking tool. Using Pyright, Pylance has the ability to supercharge your Python IntelliSense experience with rich type information, helping you write better code faster. |
| [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) | No | Automatically creates a new docstring with all arguments, their types and their default values (if defined in the function head). You just need to fill the descriptions. |

To make full use of VSCode, you can add the following lines to your `settings.json` of VSCode:

```json
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "autoDocstring.docstringFormat": "numpy",
```

The first entry here sets the automated python formatter to use. Like in Umami, you can set this to `black` to have your code auto-formatted. The second
entry enables auto-format on save. So everytime you save, `black` will format your code (style-wise). The third entry set the docstring style used in the
[Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring). Just press `Ctrl + Shift + 2` (in Linux) below
a function header and the generator will generate a fresh docstring with all arguments, their types and their default values (if defined in the function head) in the `numpy` docstring style (which is used in Umami).

## VSCode Debugger
There are plenty of tutorials and instructions for the VSCode debugger.
However, you might run into trouble when trying to debug a script which is using Umami, with the debugger telling you it can not locate the `umami` package.
In this case, try adding the directory where umami is located to the environment variables that are loaded with a new debugger terminal.

Click on `create a launch.json` as explained [here](https://code.visualstudio.com/docs/python/debugging), select the directory where you want to store it (in case you have multiple folders open) and select "Python File".
VSCode will create the default configuration file for you (located in `.vscode`). All you have to do is adding the following to the `configurations` section:
```json
            "env": {
                "PYTHONPATH": "<your_umami_dir>"
            }
```
Afterwards, the debugger should find the umami package.
