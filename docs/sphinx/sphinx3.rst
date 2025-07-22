Sphinx and RST 3: Hello World
=============================

There are two ways to approach Sphinx and RST. The first is to learn how Sphinx generates documentation from RST files first (top-down). The second is to learn how to write RST files first (bottom-up). Because I am `against top-down <https://www.change.org/p/nus-president-prof-tan-eng-chye-nus-reverse-the-mergers-and-nomoretopdown>`_ methods, let's first dicuss what RST is and how to write RST files.
As discussed during in :ref:`motivation`, RST is a markup language, similar to HTML and Markdown. While less popular than Markdown, RST is more powerful and offers more customization. For example, you can create tables, footnotes, and even math equations in RST, natively.
To create a new RST file, create a new file with the extension .rst. For example, if you want to create a new file called "hello.rst", you can do so by running the following command in your terminal:

Titles and Headers
------------------

.. code-block:: bash

  touch hello.rst

Open it up and you will see that there is nothing in the file. Unliked Markdown, RST does not have a standard way to create headers. Instead, you can create headers by underlining texts with a series of characters. For example, you might use the following scheme to create headers:

.. code-block:: rst

  Hello World
  ===========

  This is a subtitle
  ------------------

  This is a subsubtitle
  ~~~~~~~~~~~~~~~~~~~~~

Or you can use the following scheme:

.. code-block:: rst

  Hello World
  ~~~~~~~~~~~

  This is a subtitle
  ##################

  This is a subsubtitle
  ^^^^^^^^^^^^^^^^^^^^^

The only two requirements are that the header underline must be at least as long as the header text and the underline is directly under the header text.
In fact, RST, similar to Python, is a whitespace-sensitive language. This means that the number of spaces and line breaks you use matter. For example, the following RST code will not render correctly (note the erraneous whitespacing):

.. code-block:: rst

  Hello World

  ===========
