# Hierarchical Text Classificiation
### 1. Project Goal
In this repository, you will find the work that I have done to create a tool-kit that is able to place fed-in articles in one of many hierarchical classes/nodes. By a *hierarchy*, I mean that there exists a collection of nested classifications that describes where an article can fall under; for example, a news article can have the following sequence of classifications: 1. "Sports", 2. "Football", 3. "Australian Rules Football" whereas something like 1. "Sports" 2. "Hair Beauty", 3. "US Stock News" would be non-sensical. 

**The job of the tools contained in this repo**. is to *determine* the sequence of classes for any article the user provides where the allowed classes and their allowed order/organization is determined by a pre-defined taxonomy described in the `./data/` sub-directory of this repository (quick-access link here). 

Complicating matters a tiny bit is that since the world (and thus journalism) is not completely black and white, articles can have multiple class sequences such as 1. "Automotive" 2. "Auto Technology" **and** 1. "Automotive" 2."New Cars". For this reason, this system will return multiple sequences when it deems appropriate. Contrast this with the now seemingly simplicity of the typical 1-label, binary (or even multi-class) classification problem described in introductory Machine Learning courses; hence the inspiration for this project.

### 2. Repository Organization
*(Use drop-down menus to see more information)*
<details>
<summary>1. <a href="https://github.com/gosebastian12/text_hierarchical_classification/tree/master/20_newsgroup">"20_newgroup"</a>: Initial attempt to build a successful system w/simplier data.</summary>
  <ol>
    <li><code>data/</code>: Directory where tools that were created to access and organize the data used for this simplier system live.</li>
      <ol>
        <li><code>raw_data/</code>: Directory where the data was ultimately stored.</li>
        <li><code>1_Load_in_data.ipynb</code>: Code that accessed and manipulated this simplier data.</li>
      </ol>
  </ol>
</details>

<details>
<summary>2. <a href="https://github.com/gosebastian12/text_hierarchical_classification/tree/master/data">"data"</a>: Directory where propriety data gets stored.</summary>
  <ol>
    <li><code>final/</code>: Directory where data that was obtained by cleaning and/or manipulating the raw data but will be used in the final stage of development gets stored.</li>
      <ol>
        <li><code>BOW_data</code>Directory where the numerical data that was obtained by applying a Bag-of-word (BOW) transformation to the raw data lives. This data will be used with the final models of this tool-kit.</li>
      </ol>
    <li><code>interim/</code>: Directory where data that was obtained by cleaning and/or manipulating the raw data but will not be used in the final stage of development gets stored.</li>
      <ol>
        <li><code>table_dataframes</code>: Directory where pickled Pandas DataFrames that contain the data obtained via the Hindsight database that was then organized, wrangled, and used to generate additional data gets stored.</li>
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

<details>
<summary>3. <a href="https://github.com/gosebastian12/text_hierarchical_classification/tree/master/notebooks">"notebooks"</a>: Directory where notebooks that implement the modules created in the src directory live.</summary>
  <ol>
    <li><code>1_Taxonomy_Exploration.ipynb</code>: Jupyter Notebook that</li>
    <li><code>2_Accessing_Database.ipynb</code>: Jupyter Notebook that</li>
    <li><code>3_Data_Retrival.ipynb</code>: Jupyter Notebook that</li>
    <li><code>4_EDA.ipynb</code>: Jupyter Notebook that</li>
    <li><code>5_Model_Implementation.ipynb</code>: Jupyter Notebook that</li>
  </ol>
</details>

<details>
<summary>4. <a href="https://github.com/gosebastian12/text_hierarchical_classification/tree/master/src">"src"</a>: Directory that contains all of the Python code that makes up the tools implemented in this project lives.</summary>
  <ol>
    <li><code>data_scripts</code>: Sub-directory that stores all of the scripts dedicated to obtaining and manipulating data.</li>
      <ol>
        <li><code>cursor_conn_setup.py</code>: (**Not** tracked in repo. for security purposes) Python script that sets up the connection between this local script and the Hindsight AWS database.</li>
        <li><code>data_cleaning.py</code>: Python script that contains all of the functions written to clean up the text data to prepare it for a BOW transformation.</li>
        <li><code>database_compiler.py</code>: Python script that contains all of the functions written to both obtain data from the Hindsight AWS Database and then clean it.</li>
        <li><code>feature_engineering.py</code>: Python script that performs all of the numerical transformations to the prepared text data.</li>
      </ol>
    <li><code>model_scripts</code>: Sub-directory that stores all of the scripts dedicated to building the predictive models that make up this tool-kit.</li>
      <ol>
        <li><code>model_building.py</code>: Python script that contains the tools that build the several ML-models that comprise this project's toolkit.</li>
        <li><code>model_evaluation.py</code>: Python script that contains the tools that allow the user to evaluate the models that were created.</li>
        <li><code>predict.py</code>: Python script that allows the user to use all of the tools in this project to make predictions on new articles of interest.</li>
      </ol>
    <li><code>visualization_scripts</code>: Sub-directory that stores all of the scripts dedicated to creating the visualzations that summarize the results of this project.</li>
      <ol>
        <li><code>EDA.py</code>: Python script that contains the easy-to-use tools that allow the user to quickly render interesting visualizations that are relevent to this project.</li>
      </ol>
  </ol>
</details>

<details>
<summary>5. <a href="https://github.com/gosebastian12/text_hierarchical_classification/tree/master/visualizations">"visualizations"</a>: Directory that stores noteable visualizations that were created in this project.</summary>
  <ol>
    <li><code>dataset_distributions</code>: Sub-directory that contains all of the visualizations that show how the data contained in each class node is distributed.</li>
    <li><code>1_Simple_Hierarchy_Example.jpg</code>: Image that is used in the <a href="https://github.com/gosebastian12/text_hierarchical_classification/blob/master/notebooks/4_EDA.ipynb">4_EDA.ipynb</a> notebook for illustrative purposes.</li>
  </ol>
</details>

### 3. Executive Summary
To complete the goals of this project...

### 4. Useful Resources
The general resources that have been instrumental and quite useful are the following:
1. [The Hitchhiker’s Guide to Hierarchical Classification](https://towardsdatascience.com/https-medium-com-noa-weiss-the-hitchhikers-guide-to-hierarchical-classification-f8428ea1e076).
2. [Large Scale Hierarchical Classification: Foundations, Algorithms and Applications](https://cs.gmu.edu/~mlbio/presentation_SDM.pdf).
3. [Text Classification in Python](https://towardsdatascience.com/text-classification-in-python-dd95d264c802).
4. [Want to be a “real world” Data Scientist?](https://towardsdatascience.com/want-to-be-a-real-world-data-scientist-make-these-changes-to-your-portfolio-projects-e61d1139c018).
5. [How to write a production-level code in Data Science?](https://towardsdatascience.com/how-to-write-a-production-level-code-in-data-science-5d87bd75ced).
6. [A Comprehensive Guide to Understand and Implement Text Classification in Python](https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/).
7. [Cookiecutter Data Science — Organize your Projects — Atom and Jupyter](https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e).

Other more specific resources are listed in the docstrings (under the `References` section for each) for the functions written in the module `./src/`. There are also other `README` files in the important directories that can be consulted for more information on the purpose of everything in the repository.

### 5. Future Directions
