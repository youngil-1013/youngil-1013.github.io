Sphinx and RST 1: Introduction and Setting up Sphinx
====================================================

.. role:: bash(code)
   :language: bash

.. _motivation:

Motivation
----------
During the first year of my computer science journey at Yale-NUS College, 
I met Professor Olivier Danvy, who would always write his lecture notes in Sphinx, a documentation generator that coverts RST files into HTML. 
Following his footsteps, I decided to build my own website in Sphinx and RST and deploy the website using GitHub Pages.
Because this blog is about how I built the website and the nested blog in it, this blog a self-referential blog.
    
  `"This is not a self-referential sentence." <https://en.wikipedia.org/wiki/Liar_paradox>`_

RST and Sphinx
--------------
So what is this RST (or ReST) and Sphinx? RST is a markup language (think HTML and Markdown) that is used to write documents and define their formats.

Sphinx is a tool that takes markup files (RST and/or Markdown) and converts them into HTML, PDF, and other formats. 
Sphinx was originally built for RST, but as Markdown became more popular, they added support for it too.

In this blog, I will go through how I have set up Sphinx and RST to build this website. This study follows `CodeRefinery team's documentation <https://coderefinery.github.io/documentation/>`_.

.. _setupfor-ubuntu:

Sphinx setup for Ubuntu
-----------------------
The full prerequisites can be found in the `Sphinx documentation <https://www.sphinx-doc.org/en/master/usage/installation.html>`_.
However, if you have a fresh Ubuntu installation, follow these steps

* Install Python-full using apt (Ubuntu's package manager). This installation is global and will affect your system.

  .. code-block:: bash
    
    sudo apt install python3-full

* Create an umbrella folder for your project and navigate to it (The name is arbitrary).

  .. code-block:: bash
    
    mkdir mywebsites
    cd mywebsites

* Create a Python virtual environment. This will create a folder called `myenv` in your current directory (Once again, the name is arbitrary).

  .. code-block:: bash
    
    python3 -m venv myenv

* Activate the virtual environment

  .. code-block:: bash
    
    source myenv/bin/activate

* Install Sphinx using pip (add an optional :bash:`--break-system-packages` to the end of the command if you see an :bash:`error: externally-managed-environment`) after running the command below.

  .. code-block:: bash
    
    pip install -U sphinx

* Deactivate the virtual environment whenever you are done editting your website.

  .. code-block:: bash
    
    deactivate

Creating the Python virtual environment is important because it allows you to install Sphinx and its dependencies without affecting the rest of your system. Furthermore, some packages may not be available in the Ubuntu package manager, so you may need to install them using pip. 

To make sure the installation was successful, let us create a temporary Sphinx project. **Start the virtual environment** and run the following command:

.. code-block:: bash

  mkdir tmp
  cd tmp
  sphinx-quickstart

The quickstart will ask you a series of questions to set up your project.

.. code-block:: bash

  > Separate source and build directories (y/n) [n]: <hit enter>
  > Project name: <your project name>
  > Author name(s): <your name>
  > Project release []: 0.1
  > Project language [en]: <hit enter>

And a whole bunch of files will be created for you. Now, you can build your project using the following command:

.. code-block:: bash

  sphinx-build . _build

Which output an :bash:`index.html` file in the `_build` directory. You can directly open this file through your browser to see the page or you can run the following command:

.. code-block:: bash
  
  xdg-open _build/index.html

and the page will be hosted on a local server (mine is at http://127.0.0.1:8000/index.html, but yours may be different. Check your terminal for the correct address). But this requires us to build and open the page every time we make a change. To automate this process, we can use the `sphinx-autobuild` package.

.. code-block:: bash

  pip install sphinx-autobuild
  sphinx-autobuild tmp tmp/_build/html

Note that the first argument is the source directory and the second argument is the build directory. Everytime a change is made to the source directory, the build directory will be updated automatically.

As a sanity check, your :bash:`mywebsites` directory should look like this:

.. code-block:: bash

  mywebsites/
  ├── myenv/
  └── tmp/
      ├── _build/
      ├── conf.py
      ├── index.rst (This is your main file)
      ├── make.bat
      ├── Makefile
      ├── _static/
      └── _templates/