out_domain_pandas_generalization = {
1:
"""
Table epic_games, columns = [id, name, game_slug, price, release_date, platform, description, developer, publisher, genres]
Task: "Find one game platform which has one different property compared to all the other platforms."
Solution:
	    ```
		import pandas as pd

        df = pd.read_csv("epic_games.tsv", delimiter="\\t")

        windows_games = df[df['platform'] == 'Windows']
        other_platforms = df[df['platform'] != 'Windows']

        windows_games_agg = windows_games.groupby('developer').agg({'price':'mean'})
        other_platforms_agg = other_platforms.groupby('developer').agg({'price':'mean'})
	    ```
Answer:
Let’s think step by step.
"windows_games = df[df['platform'] == 'Windows']" is using 'platform' which is the column asked in the task part "Find one game platform" so it's not need to be replaced. The fixed value 'Windows' isn't explictly derived from the task so it would be replaced by the placeholder '<VALUE1>'.
"other_platforms = df[df['platform'] != 'Windows']" is using 'platform' which is the column asked in the task so it's not need to be replaced. The fixed value 'Windows' already need to be replaced by '<VALUE1>'.
"windows_games_agg = windows_games.groupby('developer').agg({'price':'mean'})" is using the column 'developer' which isn't explictly derived from the task so it would be replaced by the placeholder '<COL1>'.
Also the column 'price' isn't explictly derived from the task so it would be replaced by the placeholder '<AGG_COL1>'. The aggregation function 'mean' isn't explictly derived from the task so it would be replaced by the placeholder '<AGG_FUNC1>'.
"other_platforms_agg = other_platforms.groupby('developer').agg({'price':'mean'})" is using the column 'developer' which already replaced by '<COL1>'.
Also the column 'price' already replaced by placeholder '<AGG_COL1>'. The aggregation function 'mean' already replaced by the placeholder '<AGG_FUNC1>'.
So the replacements are: {'Windows':'<VALUE1>','developer':'<COL1>','price':'<AGG_COL1>','mean':'<AGG_FUNC1>'}
Overall the code would be replaced by the following:
	    ```
		import pandas as pd

        df = pd.read_csv("epic_games.tsv", delimiter="\\t")

        windows_games = df[df['platform'] == '<VALUE1>']
        other_platforms = df[df['platform'] != '<VALUE1>']

        windows_games_agg = windows_games.groupby('<COL1>').agg({'<AGG_COL1>':'<AGG_FUNC1>'})
        other_platforms_agg = other_platforms.groupby('<COL1>').agg({'<AGG_COL1>':'<AGG_FUNC1>'})
	    ```
""",
    2:
    """
Table ds_salaries, columns = [employee_id, work_year, experience_level, employment_type, job_title, salary, salary_currency, salary_in_usd, employee_residence, remote_ratio, company_location]
Task: "Investigate what makes data scientists to earn above the 90th percentile salary (above $219,000, according to this dataset) and drill down to a specific reason."
Solution:
	    ```
		import pandas as pd

        df = pd.read_csv("ds_salaries.tsv", delimiter="\\t")

        greater_than_219000 = df[df['salary_in_usd'] > 219000]

        job_title_counting = greater_than_219000.groupby('job_title').agg({'employee_id':'count'})

        max_job_title = job_title_counting['employee_id'].idxmax()

        max_job_title_employees = greater_than_219000[greater_than_219000['job_title'] == max_job_title]

        max_job_title_level_average = max_job_title_employees.groupby('experience_level').agg({'salary':'mean'})
	    ```
Answer:
Let’s think step by step.
"greater_than_219000 = df[df['salary_in_usd'] > 219000]" is using 'salary_in_usd' column which is explicitly derived by the task part "earn...above $219,000" so it's not need to be replaced. The fixed value '219000' also explictly derived by the task so not need to be replaced.
"job_title_counting = greater_than_219000.groupby('job_title').agg({'employee_id':'count'})" is using the column 'job_title' which isn't explictly derived from the task so it needs to be replaced by '<COL1>'. The column 'employee_id' also isn't derived from the task so it needs to be replaced by '<COL2>'.
The aggregation function 'count' isn't derived from the task so it would be replaced by the placeholder '<AGG_FUNC1>'.
"max_job_title = job_title_counting['employee_id'].idxmax()" is using the column 'employee_id' which is already replaced by the placeholder '<AGG_COL1>'. The function '.idxmax()' is instructed to not be replaced.
"max_job_title_employees = greater_than_219000[greater_than_219000['job_title'] == max_job_title]" is using the column 'job_title' which already replaced by '<COL1>'. max_job_title is a variable and not fixed value so it's not need to be replabed.
"max_job_title_level_average = max_job_title_employees.groupby('experience_level').agg({'salary':'mean'})" is using the column 'experience_level' which isn't derived from the task so it needs to be replaced by '<COL2>'. The column 'salary' also isn't derived from the task so it needs to be replaced by '<AGG_COL2>'.
The aggregation function 'mean' isn't derived from the task so it would be replaced by the placeholder '<AGG_FUNC2>'.
So the replacements are: {'job_title':'<COL1>','employee_id':'<AGG_COL1>','count':'<AGG_FUNC1>','experience_level':'<COL2>','salary':'<AGG_COL2>','mean':'<AGG_FUNC2>'}
Overall the code would be replaced by the following:
	    ```
		import pandas as pd

        df = pd.read_csv("ds_salaries.tsv", delimiter="\\t")

        greater_than_219000 = df[df['salary_in_usd'] > 219000]

        job_title_counting = greater_than_219000.groupby('<COL1>').agg({'<AGG_COL1>':'<AGG_FUNC1>'})

        max_job_title = job_title_counting['<AGG_COL1>'].idxmax()

        max_job_title_employees = greater_than_219000[greater_than_219000['<COL1>'] == max_job_title]

        max_job_title_level_average = max_job_title_employees.groupby('<COL2>').agg({'<AGG_COL2>':'<AGG_FUNC2>'})
	    ```
""",
3:
"""
Table intel_processors, columns = [Product, Status, Release Date, Cores, Threads, Lithography, Max. Turbo Freq, Base Freq, TDP, Cache]
Task: "Compare some three different subsets of products according to some properties."
Solution:
	    ```
		import pandas as pd

        df = pd.read_csv("intel_processors.tsv", delimiter="\\t")

        first_product = df[df['Product'] == 'P5341']
        second_product = df[df['Product'] == 'P5342']
        third_product = df[df['Product'] == 'P5343']

        first_product_max_cores_per_status = first_product.groupby('Status').agg({'Cores','max'})
        second_product_max_cores_per_status = second_product.groupby('Status').agg({'Cores','max'})
        third_product_max_cores_per_status = third_product.groupby('Status').agg({'Cores','max'})
	    ```
Answer:
Let’s think step by step.
"first_product = df[df['Product'] == 'P5341']" is using 'Product' column which is derived by the task part "subsets of products" so it's not need to be replaced. The fixed value 'P5341' isn't derived by the task so it would be replaced by placeholder '<VALUE1>'.
"second_product = df[df['Product'] == 'P5342']" is using 'Product' column which is derived by the task part "subsets of products so it's not need to be replaced. The fixed value 'P5342' isn't derived by the task so it would be replaced by placeholder '<VALUE2>'.
"third_product = df[df['Product'] == 'P5343']" is using 'Product' column which is derived by the task part "subsets of products so it's not need to be replaced. The fixed value 'P5343' isn't derived by the task so it would be replaced by placeholder '<VALUE3>'.
"first_product_max_cores_per_status = first_product.groupby('Status').agg({'Cores','max'})" is using the column 'Status' isn't derived by the task so it would be replaced by placeholders '<COL1>'. The column 'Cores' isn't derived by the task so it would be replaced by placeholders '<AGG_COL1>'.
 The aggregation function 'max' isn't derived from the task so it would be replaced by the placeholder '<AGG_FUNC1>'.
"second_product_max_cores_per_status = second_product.groupby('Status').agg({'Cores','max'})" is using the column 'Status','Cores' which are already replaced by placeholders '<COL1>','<AGG_COL1>' respectivly. The aggregation function 'max' is already replaced by the placeholder '<AGG_FUNC1>'.
"third_product_max_cores_per_status = third_product.groupby('Status').agg({'Cores','max'})" is using the columns 'Status','Cores' which are already replaced by placeholders '<COL1>','<AGG_COL1>' respectivly. The aggregation function 'max' is already replaced by the placeholder '<AGG_FUNC1>'.
So the replacements are: {'P5341':'<VALUE1>','P5342':'<VALUE2>','P5343':'<VALUE3>','Status':'<COL1>','Cores':'<AGG_COL1>','max':'<AGG_FUNC1>'}
Overall the code would be replaced by the following:
	    ```
		import pandas as pd

        df = pd.read_csv("intel_processors.tsv", delimiter="\\t")

        first_product = df[df['Product'] == '<VALUE1>']
        second_product = df[df['Product'] == '<VALUE2>']
        third_product = df[df['Product'] == '<VALUE3>']

        first_product_max_cores_per_status = first_product.groupby('<COL1>').agg({'<AGG_COL1>','<AGG_FUNC1>'})
        second_product_max_cores_per_status = second_product.groupby('<COL1>').agg({'<AGG_COL1>','<AGG_FUNC1>'})
        third_product_max_cores_per_status = third_product.groupby('<COL1>').agg({'<AGG_COL1>','<AGG_FUNC1>'})
	    ```
""",
    4:
"""
Table houses, columns = [Area, BHK, Bedrooms, Bathroom, Furnishing, Locality, Parking, Price, Status, Transaction, Type, Per_Sqft]
Task: "show the average cost of some two different subsets of houses."
Solution:
	    ```
        df = pd.read_csv("houses.tsv", delimiter="\\t")

        first_subset = df[df['Area'] == '7420']
        first_subset_agg = first_subset.groupby('Bedrooms').agg({'Price': 'mean'})

        second_subset = df[df['Area'] == '8960]
        second_subset_agg = second_subset.groupby('Bedrooms').agg({'Price': 'mean'})
	    ```
Answer:
Let’s think step by step.
"first_subset = df[df['Area'] == '7420']" is using 'Area' column which is not derived by the task so it would be replaced by the placeholder '<COL1>'. The fixed value '7420' is not derived by the task so it would be replaced by the placeholder '<VALUE1>'.
"first_subset_agg = first_subset.groupby('Bedrooms').agg({'Price': 'mean'})" is using 'Bedrooms' column which is not derived by the task so it would be replaced by the placeholder '<COL2>'. The column 'Price' is derived by the task part "average cost" so it won't be replaced. 
The aggregation function 'mean' is derived by the task part "average cost" so it won't be replaced.
"second_subset = df[df['Area'] == '8960']" is using 'Area' column which is already replaced by the placeholder '<COL1>'. The fixed value '8960' is not derived by the task so it would be replaced by the placeholder '<VALUE3>'.
"second_subset_agg = second_subset.groupby('Bedrooms').agg({'Price': 'mean'})" is using 'Bedrooms' column which is already replaced by the placeholder '<COL2>'. The column 'Price' is derived by the task part "average cost" so it won't be replaced. 
The aggregation function 'mean' is derived by the task part "average cost" so it won't be replaced.
So the replacements are: {'Area':'<COL1>','7420':'<VALUE1>','Bedrooms':'<COL2>','8960':'<VALUE1>'}
Overall the code would be replaced by the following:
	    ```
        df = pd.read_csv("houses.tsv", delimiter="\\t")

        first_subset = df[df['<COL1>'] == '<VALUE1>']
        first_subset_agg = first_subset.groupby('<COL2>').agg({'Price': 'mean'})

        second_subset = df[df['<COL1>'] == '<VALUE2>]
        second_subset_agg = second_subset.groupby('<COL2>').agg({'Price': 'mean'})
	    ```
""",
    5:
    """
Table emojis, columns = [Hex, Rank, Emoji, Year, Category, Subcategory, Name]
Task: "show two properties of emojis published in 2022 compared to all the emojis."
Solution:
	    ```
		import pandas as pd

        df = pd.read_csv("emojis.tsv", delimiter="\\t")

        category_count = df.groupby('Category').agg({'Hex','count'})
        subcategory_count = df.groupby('Subcategory'>).agg({'Hex','count'})

        2022_emojis = df[df['Year'] == 2022]
        2022_category_count = 2022_emojis.groupby('Category').agg({'Hex','count'})
        2022_subcategory_count = 2022_emojis.groupby('Subcategory'>).agg({'Hex','count'})
	    ```
Answer:
Let’s think step by step.
"category_count = df.groupby('Category').agg({'Hex','count'})" is using 'Category' column which is not derived by the task so it would be replaced by the placeholder '<COL1>'>. The column 'Hex' which is not derived by the task so it would be replaced by the placeholder '<AGG_COL1>'>
The aggregation function 'count' is not derived by the task so it would be replaced by the placeholder '<AGG_FUNC1>'.
"subcategory_count = df.groupby('Subcategory'>).agg({'Hex','count'})" is using 'Subcategory' column which is not derived by the task so it would be replaced by the placeholder '<COL2>'>. The column 'Hex' is already replaced by the placeholder '<AGG_COL1>'>
The aggregation function 'count' is already replaced by the placeholder '<AGG_FUNC1>'.
"2022_emojis = df[df['Year'] == 2022]" is using 'Year' column which is derived by the task part "published in 2022" so it's not need to be replaced. The fixed value '2022' is derived by the task as well so it's not need to be replaced.
"2022_category_count = 2022_emojis.groupby('Category').agg({'Hex','count'})" is using 'Category' column which is already replaced by the placeholder '<COL1>'>. The column 'Hex' is already replaced by the placeholder '<AGG_COL1>'>
The aggregation function 'count' is already replaced by the placeholder '<AGG_FUNC1>'.
"2022_subcategory_count = 2022_emojis.groupby('Subcategory'>).agg({'Hex','count'})" is using 'Subcategory' column which is already replaced by the placeholder '<COL2>'>. The column 'Hex' is already replaced by the placeholder '<AGG_COL1>'>
The aggregation function 'count' is already replaced by the placeholder '<AGG_FUNC1>'.
So the replacements are: {'Category':'<COL1>','Hex':'<AGG_COL1>','count':'<AGG_FUNC1>','Subcategory':'<COL2>'}
Overall the code would be replaced by the following:
	    ```
		import pandas as pd

        df = pd.read_csv("emojis.tsv", delimiter="\\t")

        category_count = df.groupby('<COL1>').agg({'<AGG_COL1>','<AGG_FUNC1>'})
        subcategory_count = df.groupby('<COL2>'>).agg({'<AGG_COL1>','<AGG_FUNC1>'})

        2022_emojis = df[df['Year'] == 2022]
        2022_category_count = 2022_emojis.('<COL1>').agg({'<AGG_COL1>','<AGG_FUNC1>'})
        2022_subcategory_count = 2022_emojis.groupby('<COL2>'>).agg({'<AGG_COL1>','<AGG_FUNC1>'})
	    ```
""",
    6:
"""
Table cars, columns = [addref, city, assembly, body, make, model, year, engine, transmission, fuel]
Task: "explore three different car makes in different ways."
Solution:
	    ```
		import pandas as pd

        df = pd.read_csv("cars.tsv", delimiter="\\t")

        model1 = df[df['make'] == 'Hyundai']
        model2 = df[df['make'] == 'Volkswagen']
        model3 = df[df['make'] == 'Ford']

        model1_properties = model1.groupby('model').agg({'fuel':'max'})
        model2_properties = model2.groupby('year').agg({'engine':'count'})
        model3_properties = model3.groupby('city').agg({'transmission':'mean'})
	    ```
Answer:
Let’s think step by step.
"model1 = df[df['make'] == 'Hyundai']" is using 'make' column which is derived by the task part "different car makes" so it doesn't need to be replaced. The fixed value 'Hyundai' which is not derived by the task so it would be replaced by the placeholder '<VALUE1>'.
"model2 = df[df['make'] == 'Volkswagen']" is using 'make' column which is derived by the task so it doesn't need to be replaced. The fixed value 'Volkswagen' which is not derived by the task so it would be replaced by the placeholder '<VALUE2>'.
"model3 = df[df['make'] == 'Ford']" is using 'make' column which is derived by the task so it doesn't need to be replaced. The fixed value 'Ford' which is not derived by the task so it would be replaced by the placeholder '<VALUE3>'.
"model1_properties = model1.groupby('model').agg({'fuel':'max'})" is using 'model' column which is not derived by the task so it would be replaced by the placeholder '<COL1>'. The column 'fuel' is not derived by the task so it would be replaced by the placeholder '<AGG_COL1>'.
The aggregation function 'max' is not derived by the task so it would be replaced by the placeholder '<AGG_FUNC1>'.
"model2_properties = model2.groupby('year').agg({'engine':'count'})" is using 'year' column which is not derived by the task so it would be replaced by the placeholder '<COL2>'. The column 'engine' is not derived by the task so it would be replaced by the placeholder '<AGG_COL2>'.
The aggregation function 'count' is not derived by the task so it would be replaced by the placeholder '<AGG_FUNC2>'.
"model3_properties = model3.groupby('city').agg({'transmission':'mean'})" is using 'city' column which is not derived by the task so it would be replaced by the placeholder '<COL3>'. The column 'transmission' is not derived by the task so it would be replaced by the placeholder '<AGG_COL3>'.
The aggregation function 'mean' is not derived by the task so it would be replaced by the placeholder '<AGG_FUNC3>'.
So the replacements are: {'Hyundai':'<VALUE1>','Volkswagen':'<VALUE2>','Ford':'<VALUE3>','model':'<COL1>','fuel':'<AGG_COL1>','max':'<AGG_FUNC1>','year':'<COL2>','engine':'<AGG_COL2>','count':'<AGG_FUNC2>','city':'<COL3>','transmission':'<AGG_COL3>','mean':'<AGG_FUNC3>'}
Overall the code would be replaced by the following:
	    ```
		import pandas as pd

        df = pd.read_csv("cars.tsv", delimiter="\\t")

        model1 = df[df['make'] == '<VALUE1>']
        model2 = df[df['make'] == '<VALUE2>']
        model3 = df[df['make'] == '<VALUE3>']

        model1_properties = model1.groupby('<COL1>').agg({'<AGG_COL1>':'<AGG_FUNC1>'})
        model2_properties = model2.groupby('<COL2>').agg({'<AGG_COL2>':'<AGG_FUNC2>'})
		model3_properties = model3.groupby('<COL3>').agg({'<AGG_COL3>':'<AGG_FUNC3>'})
	    ```	
""",
    8:
"""
Table spotify, columns = [Artist, Streams, Daily, As lead, Solo, As feature]
Task: "explore the data, make sure to address two interesting properties of Drake's streams."
Solution:
	    ```
		import pandas as pd

        df = pd.read_csv("spotify.tsv", delimiter="\\t")

        drake_songs = df[df['Artist'] == 'Drake']

        drake_songs_properties_1 = drake_songs.groupby('As lead').agg({'Streams':'mean'})
        drake_songs_properties_2 = drake_songs.groupby('As feature').agg({'Streams':'mean'})
	    ```
Answer:
Let’s think step by step.
"drake_songs = df[df['Artist'] == 'Drake']" is using 'Artist' column which is derived by the task part "properties of Drake's.." so it doesn't need to be replaced. The fixed value 'Drake' also derived by the task part "properties of Drake's.." so it doesn't need to be replaced.
"drake_songs_properties_1 = drake_songs.groupby('As lead').agg({'Streams':'mean'})" is using 'As lead' column which is not derived by the task so it would be replaced by the placeholer '<COL1>'. The column 'Streams' is derived by the task part "Drake's streams" so it doean't need to be replaced.
The aggregation function 'mean' is not derived by the task so it would be replaced by the placeholder '<AGG_FUNC1>'.
"drake_songs_properties_2 = drake_songs.groupby('As feature').agg({'Streams':'mean'})" is using 'As feature' column which is not derived by the task so it would be replaced by the placeholer '<COL2>'. The column 'Streams' is derived by the task part "Drake's streams" so it doean't need to be replaced.
The aggregation function 'mean' is already replaced by the placeholder '<AGG_FUNC1>'.
So the replacements are: {'As lead':'<COL1>',,'As mean':'<AGG_FUNC1>','As feature':'<COL2>'}
Overall the code would be replaced by the following:
	    ```
		import pandas as pd

        df = pd.read_csv("spotify.tsv", delimiter="\\t")

        drake_songs = df[df['Artist'] == 'Drake']

        drake_songs_properties_1 = drake_songs.groupby('<COL1>').agg({'Streams':'<AGG_FUNC1>'})
        drake_songs_properties_2 = drake_songs.groupby('<COL2>').agg({'Streams':'<AGG_FUNC1>'})
	    ```
""",
9:
"""
Table github, columns = [Name, Description, URL, Created At, Updated At, Homepage, Size, Stars, Forks, Issues]
Task: "show interesting sub-groups of five stars repositories."
Solution:
	    ```
		import pandas as pd

        df = pd.read_csv("github.tsv", delimiter="\\t")

        5_stars_repos = df[df['Stars'] == '5']

        5_stars_repos_grouped = 5_stars_repos.groupby('Created At')
        5_stars_repos_sub_grouped = 5_stars_repos_grouped.groupby('Name').agg({'Forks':'sum'})
	    ```
Answer:
Let’s think step by step.
"5_stars_repos = df[df['Stars'] == '5']" is using 'Stars' column which is derived by the task part "five stars" so it doesn't need to be replaced. The fixed value '5' also derived by the task part "five stars" so it doesn't need to be replaced.
"5_stars_repos_grouped = 5_stars_repos.groupby('Created At')" is using 'Created At' column which is not derived by the task so it would be replaced by the placeholer '<COL1>'.
"5_stars_repos_sub_grouped = 5_stars_repos_grouped.groupby('Name').agg({'Forks':'sum'})" is using 'Name' column which is not derived by the task so it would be replaced by the placeholer '<COL2>'. The column 'Forks' is not derived by the task so it would be replaced by the placeholer '<AGG_COL1>'.
The aggregation function 'sum' is already replaced by the placeholder '<AGG_FUNC1>'.
So the replacements are: {'Created At':'<COL1>','Name':'<COL2>','Forks':'<AGG_COL1>','sum':'<AGG_FUNC1>'}
Overall the code would be replaced by the following:
	    ```
		import pandas as pd

        df = pd.read_csv("github.tsv", delimiter="\\t")

        5_stars_repos = df[df['Stars'] == '5']

        5_stars_repos_grouped = 5_stars_repos.groupby('<COL1>')
        5_stars_repos_sub_grouped = 5_stars_repos_grouped.groupby('<COL2>').agg({'<AGG_COL1>':'<AGG_FUNC1>'})
	    ```
"""
}