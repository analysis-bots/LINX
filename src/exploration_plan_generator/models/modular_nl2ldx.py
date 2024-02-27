from exploration_plan_generator.clients.abstact_llm_client import AbstractLLMClient
from exploration_plan_generator.models.abstract_model import AbstractModel
from exploration_plan_generator.prompts.in_domain.flights.fligths_pandas2ldx_examples import flights_pandas2ldx_examples
from exploration_plan_generator.prompts.in_domain.netflix.netflix_pandas2ldx_examples import netflix_pandas2ldx_examples
from exploration_plan_generator.prompts.in_domain.play_store.play_store_pandas2ldx_examples import \
    play_store_pandas2ldx_examples
from exploration_plan_generator.prompts.out_domain.out_domain_pandas2ldx import out_domain_pandas2ldx_examples
from exploration_plan_generator.prompts.out_domain.out_domain_pandas_generalization import \
    out_domain_pandas_generalization

DATASET_PLACEHOLDER = "<DATASET>"
SCHEME_PLACEHOLDER = "<SCHEME>"
SAMPLE_PLACEHOLDER = "<SAMPLE>"
TASK_PLACEHOLDER = "<TASK>"
SOLUTION_PLACEHOLDER = "<SOLUTION>"
EXAMPLES_PLACEHOLDER = "<EXAMPLES>"


class ModularNL2Pd2LDX(AbstractModel):

    def __init__(self, llm_client: AbstractLLMClient):
        super().__int__(llm_client)

    def nl2ldx(self, dataset, scheme, sample, task, exclude_examples_ids, is_out_domain):
        dataset_name = dataset.split('.')[0]

        prompt = (NL2PandasStructure_prompt
                  .replace(DATASET_PLACEHOLDER, dataset)
                  .replace(SCHEME_PLACEHOLDER,str(scheme))
                  .replace(SAMPLE_PLACEHOLDER, str(sample))
                  .replace(TASK_PLACEHOLDER, task))

        pandas = self.llm_client.send_request(system_message=NL2PandasStructure_system_message, prompt=prompt).lstrip("```").lstrip("python").strip("```")

        examples = ''.join([v for k, v in out_domain_pandas_generalization.items() if k not in exclude_examples_ids])
        next_prompt = (NL2PandasContent_prompt.replace(DATASET_PLACEHOLDER, dataset).replace(SCHEME_PLACEHOLDER,str(scheme))
                       .replace(SAMPLE_PLACEHOLDER, str(sample))
                       .replace(TASK_PLACEHOLDER, task)
                       .replace(SOLUTION_PLACEHOLDER, pandas)
                       .replace(EXAMPLES_PLACEHOLDER,examples))

        pandas_generalized = self.llm_client.send_request(system_message=NL2PandasContent_system_message, prompt=next_prompt)
        if "```" in pandas_generalized: pandas_generalized = pandas_generalized[pandas_generalized.find("```") + 1:].lstrip("python").strip("```")

        ldx = self.pandas2LDX(dataset_name, pandas_generalized, exclude_examples_ids, is_out_domain)
        fixed_ldx = self.ldx_post_proccessing(ldx)

        return fixed_ldx

    def pandas2LDX(self, dataset_name, pandas, exclude_examples_ids, is_multi_domain):
        if not is_multi_domain:
            if dataset_name == "flights":
                examples = flights_pandas2ldx_examples
            elif dataset_name == "netflix":
                examples = netflix_pandas2ldx_examples
            elif dataset_name == "play_store":
                examples = play_store_pandas2ldx_examples
            else:
                raise Exception("unsupported DB")
        else:
            examples = out_domain_pandas2ldx_examples
        examples = [v for k, v in examples.items() if k not in exclude_examples_ids]

        exp_index = pandas.lower().find("explanation")
        if exp_index != -1:
            pandas = pandas[:exp_index]
        pandas = pandas.strip('\n')

        query = Pandas2LDX_prompt_prefix + ''.join(
            examples) + "\nNow convert the following while making sure '[' is closed by ']' and not by other parenthesis.\nPandas:\n" + pandas + "\nLDX:\n"
        ldx = self.llm_client.send_request(system_message=Pandas2LDX_system_message, prompt=query)
        ldx = ldx.replace(')}', ')]')
        return ldx


NL2PandasStructure_system_message = "You are an AI assistant for answering natural language analytical tasks using pandas"

NL2PandasStructure_prompt = f"""Given the dataset '{DATASET_PLACEHOLDER}' with the scheme: {SCHEME_PLACEHOLDER} and sample:\n {SAMPLE_PLACEHOLDER}:
provide pandas code for the task: {TASK_PLACEHOLDER}
Instructions:
1. you can use only: filter/groupby/agg/idxmax/idxmin.
2. supported aggregations: mean,max,min,count
3. apply aggregations only via the function agg (don't use mean(),min(),size() and etc..) and always provide aggregated column.
4. don't apply agg on ungrouped data frame.
5. don't apply multiple filters or multiple grouping or multiple aggregations in a single time

```python
"""

NL2PandasContent_system_message = "You are an AI assistant responsible for replacing unrelated expressions by placeholders"

NL2PandasContent_prompt = f"""Given task and its solution in pandas code, you need to replace by placeholder each column, fixed value or aggregation function which isn't explictly derived from the task.
Don't replace:
1. python variable names
2. logical operators like <,>,==,!=, etc.
3. functions: idxmax(),idxmin()

Are are examples:

{EXAMPLES_PLACEHOLDER}

Table {DATASET_PLACEHOLDER}, columns = [{SCHEME_PLACEHOLDER}]
Task: "{TASK_PLACEHOLDER}"
Sample:
        "{SAMPLE_PLACEHOLDER}"
        
Solution:
	    ```{SOLUTION_PLACEHOLDER}```
	    
Answer:
"""

Pandas2LDX_system_message = "You are an AI assistant for converting tasks in pandas code to LDX"

Pandas2LDX_prompt_prefix = """LDX (Language for Data Exploration) is a specification language that extends Tregex, 
a query language for tree-structured data. It allows you to partially specify structural properties of a tree, 
as well as the nodes' labels, using continuity variables (placeholders) which are determined during runtime.
The language is especially useful for specifying the order of notebook's query operations and their type and parameters.
LDX supported operators are filter (F) and groupby with aggregation (G).

Here are examples how to convert Pandas code to LDX:

Pandas:
       df = pd.read_csv("dataset.tsv", delimiter="\\t")
       average = df[<COL>].mean()
LDX:
       BEGIN CHILDREN {A1}
       A1 LIKE [G,.*,mean,<COL>] 

Pandas:       
       df = pd.read_csv("dataset.tsv", delimiter="\\t")

       do_some_operations()

       some_filter = df[df[<COL>] == <VALUE>]
LDX:
       BEGIN DESCENDANTS {A1}
       A1 LIKE [F,<COL>,eq,<VALUE>]

LDX doesn't support multiple filters/aggregations/grouping in a single time, need to split it to different operations: 

Pandas:       
       df = pd.read_csv("dataset.tsv", delimiter="\\t")

       df = df[df['column'].isin('value1','value2')]
LDX:
       BEGIN CHILDREN {A1,A2}
       A1 LIKE [F,'column',eq,'value1']
       A1 LIKE [F,'column',eq,'value2']


Pandas:       
       df = pd.read_csv("dataset.tsv", delimiter="\\t")

       df = df[df['column'].isin('value1','value2')]
       df = df.groupby('column1').agg({'column2': 'function'})
LDX:
       BEGIN CHILDREN {A1,A2}
       A1 LIKE [F,'column',eq,'value1'] and CHILDREN {B1}
        B1 LIKE [G,'column1','function,'column2']
       A2 LIKE [F,'column',eq,'value2'] and CHILDREN {B2}
        B2 LIKE [G,'column1','function,'column2']
        
Pandas:       
       df = pd.read_csv("dataset.tsv", delimiter="\\t")

       subgroups = df.groupby(['column1','column2']).agg({'column3': 'function'})
LDX:
       BEGIN CHILDREN {A1,A2}
       A1 LIKE [G,'column1','function,'column3']and CHILDREN {B1}
        B1 LIKE [G,'column2','function,'column3']

always convert idxmax/idxmin to filter on value placeholder and replace aggregate column by grouped column as follows: 

Pandas:       
      df = df.groupby('column').agg({'agg_column': 'function'})
      max_sample = df['agg_column'].idxmax()
LDX:
       BEGIN CHILDREN {A1}
       A1 LIKE [G,'column','function','agg_column'] and CHILDREN {B1}
        B1 LIKE [F,'column',eq,<VALUE>]

Pandas:       
      df = df.groupby('column').agg({'agg_column': 'function'})
      min_sample = df['agg_column'].idxmin()
LDX:
       BEGIN CHILDREN {A1}
       A1 LIKE [G,'column','function','agg_column'] and CHILDREN {B1}
        B1 LIKE [F,'column',eq,<VALUE>]   

Pandas:       
      df = df.groupby('column').agg({<AGG_COL>: 'function'})
      max_sample = df[<AGG_COL>].idxmax()
LDX:
       BEGIN CHILDREN {A1}
       A1 LIKE [G,'column','function','<AGG_COL>'] and CHILDREN {B1}
        B1 LIKE [F,'column',eq,<VALUE>]

Pandas:       
      df = df.groupby('column').agg({<AGG_COL>: 'function'})
      max_sample = df[<AGG_COL>].idxmin()
LDX:
       BEGIN CHILDREN {A1}
       A1 LIKE [G,'column','function','<AGG_COL>'] and CHILDREN {B1}
         B1 LIKE [F,'column',eq,<VALUE>]

Pandas:       
      df = df.groupby(<COL>).agg({'agg_column': 'function'})
      max_sample = df['agg_column'].idxmax()
LDX:
       BEGIN CHILDREN {A1}
       A1 LIKE [G,<COL>,'function','agg_column'] and CHILDREN {B1}
        B1 LIKE [F,<COL>,eq,<VALUE>]

Pandas:       
      df = df.groupby(<COL>).agg({'agg_column': 'function'})
      max_sample = df['agg_column'].idxmin()
LDX:
       BEGIN CHILDREN {A1}
       A1 LIKE [G,<COL>,'function','agg_column'] and CHILDREN {B1}
        B1 LIKE [F,<COL>,eq,<VALUE>]
"""
