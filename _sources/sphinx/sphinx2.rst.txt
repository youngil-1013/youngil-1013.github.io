Sphinx and RST 2: Deploying my Website on GitHub Pages
======================================================

.. role:: bash(code)
   :language: bash

GitHub Pages and Actions
------------------------

Right now, the "website" is just a bunch of HTML files in a directory, not accessible from the internet. To make it accessible, I will use GitHub Pages, which is linked with a GitHub repository.
Following the instructions will let you create a website that is accessible online, fully customizable, and free. 
This section requires a GitHub account, which you can create at `GitHub <https:github.com>`_. If you have never authorized your computer to access GitHub, you will need to set up access tokens.
Create your access token by following the instructions at `GitHub's documentation <https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token>`_.

Now, create a new repository via GitHub, and make it public (required for GitHub Pages). You can name this repository anything you want, but because this doubles as my personal website, I name it "youngil-1013.github.io".
This allows me to directly access my website at `https://youngil-1013.github.io/ <https://youngil-1013.github.io/>`_. If not, you will need to access it via `https://<your-id>.github.io/<repository-name>`.
 
This section assumes that 1\) you have a GitHub account, 2\) you have created a repository, and 3\) you have set up an access token and can access GitHub from your computer. Furthermore, it assumes that you have named your repository `<your-id>.github.io`.`
If you wish to follow CodeRefinery's instructions, you can find them at `CodeRefinery's GitHub set-up page <https://coderefinery.github.io/documentation/gh-pages/>`_. This page is heavily inspired by CodeRefinery's page, but I have added a few more details to make it easier for beginners.

Cloning the Repository and GitHub Actions Configuration
-------------------------------------------------------

  `Tip: remove all temporary files from Part 1 before proceeding`

The first step is to clone the repository to your local machine. I highly recommend to start the python virtual environment as we have done in the :ref:`setupfor-ubuntu`. You can do this by running the following command in your terminal:

.. code-block:: bash

  git clone <your-id>.github.io
  cd <your-id>.github.io

You will initially see that the repository is empty (or at best, has a README file). Now, you will need to create a Sphinx project in the repository. We have already done something similar in the previous section:

.. code-block:: bash

  mkdir docs
  cd docs
  sphinx-quickstart

docs is the directory where Sphinx will look for the RST files. You can name this directory anything you want, but for me, I have named it "docs". Let us push what we have to GitHub. I would recommend creating a .gitignore file to ignore a few files that we do not want to push to GitHub:

.. code-block:: bash

  cd ..
  echo "_build/" > .gitignore # Ignores the build directory
  echo ".vscode/" >> .gitignore # if you are using Visual Studio Code

The .ignore file is used to ignore files that you do not want to push to GitHub. Depending on which IDE you are using, you may want to add more files to the .gitignore file. Now, let us push the changes to GitHub:

.. code-block:: bash

  git add .
  git commit -m "Initial commit"
  git push

If you have an access error pop up after trying to push the changes, you may need to set up a Personal Access Token via GitHub and use the token as your password. 
Your repository should now be uploaded to `github.com/<your-id>.<your-id>.github.io`. Mine is at `https://github.com/youngil-1013/youngil-1013.github.io <https://github.com/youngil-1013/youngil-1013.github.io>`_
However, that's only the source code. We want to deploy the website. To do this, we will use GitHub Actions and Pages. First of all, you will need to create a .github directory in the root of your repository. This directory tells GitHub to run certain actions when you push to the repository.
Then, create a workflows directory inside the .github directory, which lets us set up jobs that can be triggered when certain events occur. You can either use your file explorer to create these folders or use the following bash command:

.. code-block:: bash

  mkdir .github
  mkdir .github/workflows

Now, create a file named deploy.yml in the workflows directory. This file will contain the instructions for GitHub Actions to deploy the website. You can create this file using your text editor or use the following bash command:

.. code-block:: bash

  touch .github/workflows/deploy.yml

Now, open the deploy.yml file in your text editor and paste the following code:

.. code-block:: yaml

  name: Deploy Docs

  on: [push, pull_request, workflow_dispatch]

  permissions:
    contents: write

  jobs:
    docs:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
        - name: Install dependencies
          run: |
            pip install sphinx myst-parser sphinx-toolbox sphinx_rtd_theme
        - name: Sphinx build
          run: |
            sphinx-build docs _build
        - name: Deploy to GitHub Pages
          uses: peaceiris/actions-gh-pages@v3
          if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}
          with:
            publish_branch: gh-pages
            github_token: ${{ secrets.GITHUB_TOKEN }}
            publish_dir: _build/
            force_orphan: true

Let's understand what this code does:

* The **name** of the workflow is "Deploy Docs".
  
* The workflow is triggered **on** a push, pull request, or when you manually trigger the workflow.
  
* The **permissions** are set to write (so that the workflow can write to the repository).
 
* The **job** specifies what the job actually is:
  
  * The job **runs on** the latest version of Ubuntu.
  
  * The **steps** to the job are as follows:
  
    * The repository is **checked out**.
  
    * **Python is set up**.
  
    * **Install dependencies** refers the following:
  
      * Calls *pip*, python's package manager, to install the following packages:
  
        * *sphinx* is the documentation generator.
  
        * *myst-parser* is the markdown parser for Sphinx.
  
        * *sphinx-toolbox* is a collection of Sphinx extensions.
  
        * *sphinx_rtd_theme* is the theme for Sphinx.
  
  * **Sphinx build** refers the following:

    * Calls *sphinx-build* to build the documentation in the docs directory and output it to the _build directory.

  * **Deploy to GitHub Pages** refers the following:
  
    * The action **uses** the peaceiris/actions-gh-pages@v3 action.
  
    * The action is **conditional** on the event being a push to the main branch.
  
    * The action **publishes** the _build directory to the gh-pages branch.
  
    * The action uses the **github_token** to authenticate the action.
  
    * The action **publishes** the _build directory to the gh-pages branch.
  
    * The action **force_orphan** is set to true, which means that the gh-pages branch will be overwritten every time the action is run.

During a push, pull request, or manual trigger, you can check out what the job does at the Actions tab of the repository. There are two important sections to the deploy.yml file we have written.
First, you must include all dependencies (anything you have downloaded with :bash:`pip install ...` in the Sphinx project after the pip install command or you might see the following error message in the Actions tab after clicking on the job:

.. code-block:: bash

  Traceback
  =========

        File "/opt/hostedtoolcache/Python/3.13.2/x64/lib/python3.13/site-packages/sphinx/registry.py", line xxx, in load_extension
          raise ExtensionError(
              __('Could not import extension %s') % extname, err
          ) from err
      sphinx.errors.ExtensionError: Could not import extension <package_name> (exception: No module named '<package_name>')

Sanity Testing
~~~~~~~~~~~~~~

Let's double check that your repository on GitHub has the following structure:

.. code-block:: bash

  <your-id>.github.io/
  ├── .github/
  │   └── workflows/
  │       └── deploy.yml
  ├── docs/
  │   ├── conf.py
  │   ├── index.rst
  │   ├── make.bat
  │   └── Makefile
  ├── .gitignore
  └── README.md

If not, you may need to go back and make changes to the repository. If you see any errors (Red X-marks) in the Actions tab, you can click on the error to see what went wrong. If the errors are anything other than the ones mentioned above, you are on your own here.

GitHub Pages Configuration
--------------------------

We're almost there. Right now, even if you push to the repository, you will not be able to see the website because our GitHub Action builds the website and deploys it to the gh-pages branch. Just a few more steps to go.

  `"Let perseverance finish its work so that you may be mature and complete, not lacking anything."`

Let us now tell Github where to look for the website. Go to the repository **Settings** pages and find the **Pages** section. You will see a dropdown menu that says "None". Click on this and select the gh-pages branch (If this does not appear, check the Actions tab, as there might be errors). Then, select /(root) as the source. Then, press the **Save** button. Now, all is set!

Now, push everything you have onto the repository. You can see your "website" live on the internet at `https://<your-id>.github.io/<repository-name>`. It usually takes a few seconds to a few minutes for the website to be live. If you see a 404 error even after waiting, check if the **Actions** tab has any errors. If you see an error, you can click on the error to see what went wrong.
If the errors are anything other than the ones mentioned above, you are on your own here.

Congratulations! You have successfully deployed your website using Sphinx and GitHub Pages. But to be honest, the website is a bit bleak as of now. In the next section, I will discuss how we can add more features and pages to the website.