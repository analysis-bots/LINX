play_store_nl2ldx_examples = {
    1:
    """
task: find one category which has one different property compared to all the other categories

LDX:
    BEGIN CHILDREN {A1,A2}
    A1 LIKE [F,category,eq,<VALUE>] and CHILDREN {B1}
      B1 LIKE [G,<COL>,<AGG_FUNC>,<AGG_COL>]
    A2 LIKE [F,category,ne,<VALUE>] and CHILDREN {B2}
      B2 LIKE [G,<COL>,<AGG_FUNC>,<AGG_COL>]

explanation: Split the apps to two sets - one with a certain category and one the other categories.
Then apply the same aggregation on both of them in order to compare them.
""",
    2:
    """
task: investigate what makes apps to have more than 1000000 installs and drill down to a specific reason

LDX:
    BEGIN CHILDREN {A1}
    A1 LIKE [F,installs,gt,1000000] and CHILDREN {B1,B2}
    B1 LIKE [G,<COL1>,<AGG_FUNC1>,<AGG_COL1>]
    B2 LIKE [F,<COL1>,eq,<VALUE1>] and CHILDREN {C1,C2}
        C1 LIKE [G,<COL2>,<AGG_FUNC2>,<AGG_COL2>]
        C2 LIKE [F,<COL2>,eq,<VALUE2>]

explanation: filter the apps for those with number of installs bigger than 1000000.
Then, group according to some column and apply some aggregation in order to find some column that significantly influences the distribution of those apps.
After that filter on one of the values of the selected column from the previous step. Repeat it once again to drill down more.
""",
    3:
    """
task: compare some three different subsets of content ratings according to some properties

LDX:
    BEGIN CHILDREN {A1,A2,A3}
    A1 LIKE [F,content_rating,eq,<VALUE1>] and CHILDREN {B1}
      B1 LIKE [G,<COL>,<AGG_FUNC>,<AGG_COL>]
    A2 LIKE [F,content_rating,eq,<VALUE2>] and CHILDREN {B2}
      B2 LIKE [G,<COL>,<AGG_FUNC>,<AGG_COL>]
    A3 LIKE [F,content_rating,eq,<VALUE3>] and CHILDREN {B3}
      B3 LIKE [G,<COL>,<AGG_FUNC>,<AGG_COL>]

explanation: Split the apps to three sets, each one filtered to a different content rating.
Then apply the same aggregation on each of them in order to compare them.
""",
    4:
    """
task: show the average rating of some two different subsets of apps

LDX:
    BEGIN CHILDREN {A1,A2}
    A1 LIKE [F,<COL1>,eq,<VALUE1>] and CHILDREN {B1}
      B1 LIKE [G,<AGG_COL1>,mean,rating]
    A2 LIKE [F,<COL2>,eq,<VALUE2>] and CHILDREN {B2}
      B2 LIKE [G,<AGG_COL2>,mean,rating]

explanation: filter the apps to some column and some of its values.
Then, group the apps according to some column and calculate the average rating. Do so one more time but on different subset of the apps.
""",
    5:
    """
task: show two properties of apps with app id 1 compared to all the apps

LDX:
    BEGIN CHILDREN {A1,A2,A3}
    A1 LIKE [G,<COL1>,<AGG_FUNC1>,<AGG_COL1>]
    A2 LIKE [G,<COL2>,<AGG_FUNC2>,<AGG_COL2>]
    A3 LIKE [F,app_id,eq,1] and CHILDREN {B1,B2}
      B1 LIKE [G,<COL1>,<AGG_FUNC1>,<AGG_COL1>]
      B2 LIKE [G,<COL2>,<AGG_FUNC2>,<AGG_COL2>]

explanation: Apply two aggregations. Also filter the original data to app id 1 and apply the same two aggregations in order to compare it to the previous step.
""",
    6:
    """
task: explore three different app ids in different ways

LDX:
    BEGIN CHILDREN {A1,A2,A3}
    A1 LIKE [F,category,eq,<VALUE1>] and CHILDREN {B1}
        B1 LIKE [G,<COL1>,<AGG_FUNC1>,<AGG_COL1>]
    A2 LIKE [F,category,eq,<VALUE2>] and CHILDREN {B2}
        B2 LIKE [G,<COL2>,<AGG_FUNC2>,<AGG_COL2>]
    A3 LIKE [F,category,eq,<VALUE3>] and CHILDREN {B3}
        B3 LIKE [G,<COL3>,<AGG_FUNC3>,<AGG_COL3>]

explanation: filter to three different app categories and for each one show some properties.
   """,
    8:
    """
task: explore the data, make sure to address two interesting aspects of Free apps

LDX:
    BEGIN DESCENDANTS {A1}
    A1 LIKE [F,type,eq,Free] and CHILDREN {B1,B2}
        B1 LIKE [G,<COL1>,<AGG_FUNC1>,<AGG_COL1>]
        B2 LIKE [G,<COL2>,<AGG_FUNC2>,<AGG_COL2>]

explanation: Use descendant in order to filter type to free at some point. Then, show two different properties using two different group by operations.
""",
    9:
    """
task: show interesting sub-groups of apps require 4 Android version

LDX:
    BEGIN CHILDREN {A1}
    A1 LIKE [F,min_android_ver,eq,4] and CHILDREN {B1}
        B1 LIKE [G,.*] and CHILDREN {C1}
            C1 LIKE [G,.*]

explanation: Filter to apps with 4 min android version.
Then apply some groupby to view it as interesting groups, and apply another different groupby to view interesting sub-groups.
"""
}