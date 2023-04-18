
## Workshop Hands-on Machine Learning in Python, March-April 2023 edition

by Katrien Antonio, Jonas Crevecoeur and Roel Henckaerts

Course materials for the *Hands-on Machine Learning in Python* course
(March - April 2023).

📆 Module 1 on March 7 & 14, 2023, Module 2 on March 21 & 28, 2023 and
Module 3 on April 18 & April 25, 2023 <br> ⏱ From 9.30 am to 12.30 pm
<br> 📌 online, organized by Actuarieel Instituut

Course materials will be posted in the week before the workshop. You are
now on the landing page for **Module 3: neural networks**.

## Prework

<p align="justify">

The workshop requires a basic understanding of Python. A good starting
level is the material covered in the basic Python workshop offered by
the Actuarieel Instituut [Basiscursus Python voor actuarieel
professionals](https://www.actuarieelinstituut.nl/permanente-educatie/basiscursus-python-voor-actuarieel-professionals-1.htm).

</p>

Familiarity with statistical or machine learning methods is *not*
required. The workshop gradually builds up these concepts, with an
emphasis on hands-on demonstrations and exercises.

The Google Colab environment should be up and running before coming to
the workshop. Please visit the **Software requirements** posted below.

## Overview

<p text-align="justify">

This workshop introduces the *essential concepts of building machine
learning models with Python*. Throughout the workshop you will gain
insights in the foundations of machine learning methods, including
*resampling methods*, *data preprocessing steps* and the *tuning of
parameters*. You will cover a variety of *statistical and machine
learning methods*, ranging from GLMs, over tree-based machine learning
methods to neural networks. You will acquire insights in the foundations
of these methods, learn how to set-up the model building process, and
focus on building a good understanding of the resulting model output and
predictions.

</p>

<p align="justify">

Leaving this workshop, you should have a firm grasp of the working
principles of a variety of machine learning methods and be able to
explore their use in practical settings. Moreover, you should have
acquired the fundamental insights to explore some other methods on your
own.

</p>

## Schedule and Course Material

The schedule will gradually be completed over the next days. The
detailed schedule is subject to small changes.

|  Module  |  Session  | Duration                | Description                                              | Lecture material                                                                                                              | Link to Colab                                                                                     | Link to Colab with solutions                                                                      |
| :------: | :-------: | ----------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Prework  |           | your own pace           | check the Prework and Software Requirements instructions |                                                                                                                               |                                                                                                   |                                                                                                   |
| Prework  |           | day before the workshop | download the course material from the GitHub repo        |                                                                                                                               |                                                                                                   |                                                                                                   |
| Module 3 | Session 1 | 09.30 - 10.10           | Toolbox and tensors                                      | [sheets](https://katrienantonio.github.io/hands-on-machine-learning-Python-module-3/sheets/ML_part3_Python.html#start)        | [notebook](https://colab.research.google.com/drive/1UiMLC4p0JGZvUQ_Ug7DJxY3CQQvcIjB4?usp=sharing) | [notebook](https://colab.research.google.com/drive/14A-FosOJLfTRTpBKpGVnM0P7QOhqkqIg?usp=sharing) |
|          |           | 10.10 - 10.30           | MNIST data                                               | [sheets](https://katrienantonio.github.io/hands-on-machine-learning-Python-module-3/sheets/ML_part3_Python.html#data-sets)    |                                                                                                   |                                                                                                   |
|          |           | 10.40 - 11.10           | De-mystifying neural networks                            | [sheets](https://katrienantonio.github.io/hands-on-machine-learning-Python-module-3/sheets/ML_part3_Python.html#demystify)    |                                                                                                   |                                                                                                   |
|          |           | 11.10 - 12.30           | Neural networks in {keras}                               | [sheets](https://katrienantonio.github.io/hands-on-machine-learning-Python-module-3/sheets/ML_part3_Python.html#fundamentals) |                                                                                                   |                                                                                                   |
| Module 3 | Session 2 | 9.30 - 10.10            | Claim frequency with neural nets                         | [sheets](https://katrienantonio.github.io/hands-on-machine-learning-Python-module-3/sheets/ML_part3_Python.html#regression)   |                                                                                                   |                                                                                                   |
|          |           | 10.10 - 10.30           | Adding a skip connection                                 | [sheets](https://katrienantonio.github.io/hands-on-machine-learning-Python-module-3/sheets/ML_part3_Python.html#regression)   |                                                                                                   |                                                                                                   |
|          |           | 10.30 - 11.10           | CNNs                                                     | [sheets](https://katrienantonio.github.io/hands-on-machine-learning-Python-module-3/sheets/ML_part3.html#cnn)                 |                                                                                                   |                                                                                                   |
|          |           | 11.20 - 11.30           | Autoencoders                                             | [sheets](https://katrienantonio.github.io/hands-on-machine-learning-Python-module-3/sheets/ML_part3_Python.html#autoencoder)  |                                                                                                   |                                                                                                   |
|          |           | 11.30 - 12.30           | Working on case study                                    |                                                                                                                               |                                                                                                   |                                                                                                   |

##### Module 3: Neural networks

In two sessions we cover:

  - the Python toolbox
  - tensors and operations on tensors
  - basics of feed-forward artificial neural networks
  - an architecture with layers in {keras}
  - loss function, forward pass and backpropagation
  - performance metrics
  - claim frequency and severity modelling with neural networks
  - auto encoders
  - convolutional neural networks.

Download lecture sheets in
[pdf](https://katrienantonio.github.io/hands-on-machine-learning-Python-module-3/sheets/ML_part3.pdf).

## Software Requirements

You have **two options** to join the coding exercises covered during the
workshop. Either you join the Google Colabs dedicated to the workshop,
and then you’ll execute Python code from your browser. Or you use your
local installation of Python, as distributed via the Anaconda platform.

We kindly ask participants to **join the Google Colab as default**\!

### Google Colab - default\!

Google Colaboratory, or “Colab” for short, allows you to write and
execute Python in your browser. Having a Google account you should be
able to open the notebook and to run the Python code cells. Please check
this before the workshop by opening [the following
link](https://colab.research.google.com/drive/13rQM_WVZJNfj-uRdRHsCX0MhUdH2FE-w)
to a Colab with a few basic instructions. To execute the code in a code
cell, select it with a click and then either press the play button to
the left of the code, or use the keyboard shortcut “Command/Ctrl+Enter”.
To edit the code, just click the cell and start editing. Watch
[Introduction to Colab](https://www.youtube.com/watch?v=inN8seMm7UI) to
learn more.

To open a (locally stored) notebook on Google Colab you can take the
following steps:

  - go to Google Colab: <https://colab.research.google.com>
  - click on Upload or use File \> Upload notebook and upload your local
    copy of the notebook.

In Colab many Python packages are already available by default. If a
packages is missing it can be installed as explained here. We will
explain how to do this in the first session of Module 1.

### Local installation - optional

*The Anaconda platform*

We recommend obtaining your local installation of Python via the
distribution platform Anaconda. Here you should

  - download Anaconda at
    <https://www.anaconda.com/distribution/#download-section>, select
    the version for Python 3.8 and make sure to pick the right operating
    system (top of the page: select Windows, macOS or Linux)
  - install Anaconda; this is straightforward after launching the
    installer, but (in case you are in doubt) some instructions are at
    <https://docs.anaconda.com/anaconda/install/windows/>.

*Jupyter notebook*

[Jupyter notebook](https://jupyter.org/) is a web application that
allows to edit and run notebook documents via a web browser. A notebook
allows you to combine text, images and executable code in a single
document. Jupyter notebook supports over 40 programming languages,
including Python.  
To open the notebook you have downloaded simply go as follows:

  - launch Anaconda
  - launch Jupiter Notebook from the Anaconda main screen
  - open the notebook file stored on your computer.

You can give this a try with the following `.ipynb` notebook with a few
basic instructions: [simple
notebook](https://katrienantonio.github.io/hands-on-machine-learning-Python-module-1/notebooks/read_data_module_1.ipynb).

## Instructors

<img src="img/Katrien.jpg" width="110"/>

<p align="justify">

[Katrien Antonio](https://katrienantonio.github.io/) is professor in
insurance data science at KU Leuven and associate professor at
University of Amsterdam. She teaches courses on data science for
insurance, life and non-life insurance mathematics and loss models.
Research-wise Katrien puts focus on pricing, reserving and fraud
analytics, as well as mortality dynamics.

</p>

<p align="justify">

*Jonas Crevecoeur* is a post-doctoral researcher in biostatistics at KU
Leuven. He recently obtained his PhD within the insurance research group
at KU Leuven and holds the degrees of MSc in Mathematics, MSc in
Insurance Studies and MSc in Financial and Actuarial Engineering (KU
Leuven). Before starting the PhD program he worked as an intern with QBE
Re (Belgium office) where he studied multiline products and copulas.
Jonas was a PhD fellow of the Research Foundation - Flanders (FWO, PhD
fellowship fundamental research).

</p>

<p align="justify">

*Roel Henckaerts* holds the degrees of MSc in Mathematical Engineering,
MSc in Insurance Studies and Financial and Actuarial Engineering (KU
Leuven) and PhD in Business Economics (KU Leuven). Before starting the
PhD program he worked as an intern with AIG (London office) and KBC.
Roel is PhD fellow of the Research Foundation - Flanders (FWO, PhD
fellowship strategic basic research). After the completion of his PhD,
Roel joined [Prophecy Labs](https://prophecylabs.com/): an AI/ML startup
with experience in building end-to-end data solutions that provide
concrete business value.

</p>

Happy learning\!

-----
