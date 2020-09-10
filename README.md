# Hierarchical Text Classificiation
### 1. Project Goal
In this repository, you will find the work that I have done to create a tool-kit that is able to place fed-in articles in one of many hierarchical classes/nodes. By a *hierarchy*, I mean that there exists a collection of nested classifications that describes where an article can fall under; for example, a news article can have the following sequence of classifications: 1. "Sports", 2. "Football", 3. "Australian Rules Football" whereas something like 1. "Sports" 2. "Hair Beauty", 3. "US Stock News" would be non-sensical. 

**The job of the tools contained in this repo**. is to *determine* the sequence of classes for any article the user provides where the allowed classes and their allowed order/organization is determined by a pre-defined taxonomy described in the `./data/` sub-directory of this repository (quick-access link here). 

Complicating matters a tiny bit is that since the world (and thus journalism) is not completely black and white, articles can have multiple class sequences such as 1. "Automotive" 2. "Auto Technology" **and** 1. "Automotive" 2."New Cars". For this reason, this system will return multiple sequences when it deems appropriate. Contrast this with the now seemingly simplicity of the typical 1-label, binary (or even multi-class) classification problem described in introductory Machine Learning courses; hence the inspiration for this project.

### 2. Repository Organization
<details open>
<summary>1. <a href="https://github.com/gosebastian12/text_hierarchical_classification/tree/master/20_newsgroup">"20_newgroup"</a>: Initial Attempt to build a successful system w/simplier data.</summary>
  <ol>
    <li><code>data/</code>: Tools that were created to access and organize the data used for this simplier system.</li>
      <ol>
        <li><code>raw_data/</code>: Directory where the data was ultimately stored.</li>
        <li><code>1_Load_in_data.ipynb</code>: Code that accessed and manipulated this simplier data.</li>
      </ol>
  </ol>
</details>

<details open>
<summary>2. <a href="https://github.com/gosebastian12/text_hierarchical_classification/tree/master/data">"data"</a>: Directory where propriety data gets stored.</summary>
  <ol>
    <li><code>final/</code>: Directory where data that was obtained by cleaning and/or manipulating the raw data but will be used in the final stage of development gets stored.</li>
    <li><code>interim/</code>: Directory where data that was obtained by cleaning and/or manipulating the raw data but will not be used in the final stage of development gets stored.</li>
      <ol>
        <li><code>table_dataframes</code>: This directory is where </li>
      </ol>
    <li><code>raw/</code>: Directory where the initially accessed data gets stored.</li>
      <ol>
        <li><code> annotations_tracking_Taxonomy.csv</code>: File that describes the classification taxonomy used in this project in a tabular form.</li>
        <li><code> iab_taxonomy-v2.json</code>: File that describes the classification taxonomy used in this project in a nested form.</li>
        <li><code> table_names.pkl</code>: Serialized file that contains all of the table names populated in the database that was used to access the raw data for this project. When loaded into Python, the returned object is a list.</li>
        <li>Here is where a detailed description of the taxonomy used in this project is given.</li>
      </ol>
  </ol>
</details>

<details open>
<summary>3. <a href="https://github.com/gosebastian12/text_hierarchical_classification/tree/master/notebooks">"notebooks"</a></summary>
3.1 Well, you asked for it!
</details>

<details open>
<summary>4. <a href="https://github.com/gosebastian12/text_hierarchical_classification/tree/master/src">"src"</a></summary>
4.1. Well, you asked for it!
</details>

<details open>
<summary>5. <a href="https://github.com/gosebastian12/text_hierarchical_classification/tree/master/visualizations">"visualizations"</a></summary>
5.1. Well, you asked for it!
</details>

### Useful Resources
The general resources that have been instrumental and quite useful are the following:
1. [The Hitchhiker’s Guide to Hierarchical Classification](https://towardsdatascience.com/https-medium-com-noa-weiss-the-hitchhikers-guide-to-hierarchical-classification-f8428ea1e076).
2. [Large Scale Hierarchical Classification: Foundations, Algorithms and Applications](https://cs.gmu.edu/~mlbio/presentation_SDM.pdf).
3. [Text Classification in Python](https://towardsdatascience.com/text-classification-in-python-dd95d264c802).
4. [Want to be a “real world” Data Scientist?](https://towardsdatascience.com/want-to-be-a-real-world-data-scientist-make-these-changes-to-your-portfolio-projects-e61d1139c018).
5. [How to write a production-level code in Data Science?](https://towardsdatascience.com/how-to-write-a-production-level-code-in-data-science-5d87bd75ced).
6. [A Comprehensive Guide to Understand and Implement Text Classification in Python](https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/).
7. [Cookiecutter Data Science — Organize your Projects — Atom and Jupyter](https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e).

Other more specific resources are listed in the docstrings (under the `References` section for each) for the functions written in the module `./src/`. There are also other `README` files in the important directories that can be consulted for more information on the purpose of everything in the repository.
