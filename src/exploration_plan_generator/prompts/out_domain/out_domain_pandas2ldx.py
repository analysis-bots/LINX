out_domain_pandas2ldx_examples = {
    1:
        """
Pandas:
        df = pd.read_csv("epic_games.tsv", delimiter="\\t")

        some_platform = df[df['platform'] == <VALUE>]
        other_platforms = df[df['platform'] != <VALUE>]

        some_platform_agg = some_platform.groupby(<COL>).agg({<AGG_COL>:<AGG_FUNC>})
        other_platforms_agg = other_platforms.groupby(<COL>).agg({<AGG_COL>:<AGG_FUNC>})

        # compare the two aggregations
        comparison = pd.concat([some_platform_agg, other_platforms_agg], axis=1)
LDX:
      BEGIN CHILDREN {A1,A2}
      A1 LIKE [F,platform,eq,<VALUE>] and CHILDREN {B1}
          B1 LIKE [G,<COL>,<AGG_FUNC>,<AGG_COL>]
      A2 LIKE [F,platform,ne,<VALUE>] and CHILDREN {B2}
          B2 LIKE [G,<COL>,<AGG_FUNC>,<AGG_COL>]
    """,
    2:
        """
Pandas: 
       df = pd.read_csv("ds_salaries.tsv", delimiter="\\t")
       greater_than_219000 = df[df['salary_in_usd'] > 219000]
       properties1 = greater_than_219000.groupby(<COL1>).agg({<AGG_COL1>:<AGG_FUNC1>})
       max_property = job_title_counting['<COL1>'].idxmax()
       focus_of_col1 = greater_than_219000[greater_than_219000[<COL1>] == max_property]
       properties2 = focus_of_col1.groupby(<COL2>).agg({<AGG_COL2>:<AGG_FUNC2>})
LDX:
        BEGIN CHILDREN {A1}
        A1 LIKE [F,salary_in_usd,gt,219000] and CHILDREN {B1,B2}
        B1 LIKE [G,<COL1>,<AGG_FUNC1>,<AGG_COL1>]
        B2 LIKE [F,<COL1>,eq,<VALUE1>] and CHILDREN {C1,C2}
            C1 LIKE [G,<COL2>,<AGG_FUNC2>,<AGG_COL2>]
    """,
    3:
        """
Pandas:
        df = pd.read_csv("intel_processors.tsv", delimiter="\\t")

        first_product = df[df['Product'] == <VALUE1>]
        second_product = df[df['Product'] == <VALUE2>]
        third_product = df[df['Product'] == <VALUE3>]

        first_product_agg = first_product.groupby(<COL>).agg({<AGG_COL>:<AGG_FUNC>})
        second_product_agg = second_product.groupby(<COL>).agg({<AGG_COL>:<AGG_FUNC>})
        third_product_agg = third_product.groupby(<COL>).agg({<AGG_COL>:<AGG_FUNC>})
LDX:
      BEGIN CHILDREN {A1,A2,A3}
      A1 LIKE [F,Product,eq,<VALUE1>] and CHILDREN {B1}
          B1 LIKE [G,<COL>,<AGG_FUNC>,<AGG_COL>]
      A2 LIKE [F,Product,eq,<VALUE2>] and CHILDREN {B2}
          B2 LIKE [G,<COL>,<AGG_FUNC>,<AGG_COL>]
      A3 LIKE [F,Product,eq,<VALUE3>] and CHILDREN {B3}
          B3 LIKE [G,<COL>,<AGG_FUNC>,<AGG_COL>]
    """,
    4:
        """
Pandas:       
       df = pd.read_csv("houses.tsv", delimiter="\\t")

       first_subset = df[df[<COL1>] == <VALUE1>]
       first_subset_agg = first_subset.groupby(<AGG_COL1>).agg({'Price': 'mean'})

       second_subset = df[df[<COL2] == <VALUE2>]
       second_subset_agg = second_subset.groupby(<AGG_COL2>).agg({'Price': 'mean'})

       highest_departure_delay = max(first_subset_agg['Price'], second_subset_agg['Price'])
LDX:
      BEGIN CHILDREN {A1,A2}
      A1 LIKE [F,<COL1>,eq,<VALUE1>] and CHILDREN {B1}
          B1 LIKE [G,<AGG_COL1>,mean,Price]
      A2 LIKE [F,<COL2>,eq,<VALUE2>] and CHILDREN {B2}
          B2 LIKE [G,<AGG_COL2>,mean,Price]
    """,
    5:
        """
Pandas:       
       df = pd.read_csv("emojis.tsv", delimiter="\\t")

       emojis_properties_1 = df.groupby(<COL1>).agg({<AGG_COL1>:<AGG_FUNC1>})
       emojis_properties_2 = df.groupby(<COL2>).agg({<AGG_COL2>:<AGG_FUNC2>})

       2022_emojis = df[df['Year'] == 2022]
       2022_emojis_properties_1 = 2022_emojis.groupby(<COL1>).agg({<AGG_COL1>:<AGG_FUNC1>})
       2022_emojis_properties_2 = 2022_emojis.groupby(<COL2>).agg({<AGG_COL2>:<AGG_FUNC2>})
LDX:
      BEGIN CHILDREN {A1,A2,A3}
      A1 LIKE [G,<COL1>,<AGG_FUNC1>,<AGG_COL1>]
      A2 LIKE [G,<COL2>,<AGG_FUNC2>,<AGG_COL2>]
      A3 LIKE [F,Year,eq,2022] and CHILDREN {B1,B2}
          B1 LIKE [G,<COL1>,<AGG_FUNC1>,<AGG_COL1>]
          B2 LIKE [G,<COL2>,<AGG_FUNC2>,<AGG_COL2>]
    """,
    6:
        """
Pandas:       
       df = pd.read_csv("cars.tsv", delimiter="\\t")

       model1 = df[df['model'] == <VALUE1>]
       model2 = df[df['model'] == <VALUE2>]
       model3 = df[df['model'] == <VALUE3>]

       model1_properties = model1.groupby(<COL1>).agg({<AGG_COL1>:<AGG_FUNC1>})
       model2_properties = model2.groupby(<COL2>).agg({<AGG_COL2>:<AGG_FUNC2>})
       model3_properties = model3.groupby(<COL3>).agg({<AGG_COL3>:<AGG_FUNC3>})
LDX:
        BEGIN CHILDREN {A1,A2,A3}
        A1 LIKE [F,model,eq,<VALUE1>] and CHILDREN {B1}
            B1 LIKE [G,<COL1>,<AGG_FUNC1>,<AGG_COL1>]
        A2 LIKE [F,model,eq,<VALUE2>] and CHILDREN {B2}
            B2 LIKE [G,<COL2>,<AGG_FUNC2>,<AGG_COL2>]
        A3 LIKE [F,model,eq,<VALUE3>] and CHILDREN {B3}
            B3 LIKE [G,<COL3>,<AGG_FUNC3>,<AGG_COL3>]
    """,
    8:
        """
Pandas:       
       df = pd.read_csv("spotify.tsv", delimiter="\\t")

       do_some_operations()

       drake_songs = df[df['Artist'] == 'Drake']

       drake_songs_properties_1 = drake_songs.groupby(<COL1>).agg({<AGG_COL1>:<AGG_FUNC1>})
       drake_songs_properties_2 = drake_songs.groupby(<COL2>).agg({<AGG_COL2>:<AGG_FUNC2>})
LDX:
        BEGIN DESCENDANTS {A1}
        A1 LIKE [F,Artist,eq,Drake] and CHILDREN {B1,B2}
            B1 LIKE [G,<COL1>,<AGG_FUNC1>,<AGG_COL1>]
            B2 LIKE [G,<COL2>,<AGG_FUNC2>,<AGG_COL2>]
    """,
    9:
        """
Pandas:       
        df = pd.read_csv("github.tsv", delimiter="\\t")

        5_stars_repos = df[df['Stars'] == '5']

        5_stars_repos_grouped = 5_stars_repos.groupby(<COL1>).agg({<AGG_COL1>:<AGG_FUNC1>})
        5_stars_repos_sub_grouped = 5_stars_repos_grouped.groupby(<COL2>).agg({<AGG_COL2>:<AGG_FUNC2>})
LDX:
        BEGIN CHILDREN {A1}
        A1 LIKE [F,Stars,eq,5] and CHILDREN {B1}
            B1 LIKE [G,<COL1>,<AGG_FUNC1>,<AGG_COL1>] and CHILDREN {C1}
                C1 LIKE [G,<COL2>,<AGG_FUNC2>,<AGG_COL2>]
    """
}