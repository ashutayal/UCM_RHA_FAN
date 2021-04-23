#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:01:53 2020

@author: ashutayal
"""

# Import libraries
from statsmodels.stats.contingency_tables import mcnemar as mcnemars_stat
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ipywidgets import interact, interact_manual
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['figure.dpi'] = 300

# Set working directory to location of csvs


#######
# CPD
#######

path = os.path.join(os.getcwd(), 'desktop', 'AM_csvs', 'CPD')

df_2017 = pd.read_csv(os.path.join(path, '2017-18.csv'))
df_2018 = pd.read_csv(os.path.join(path, '2018-19.csv'))
df_2019 = pd.read_csv(os.path.join(path, '2019-20.csv'))

# Verify if read correctly, view top 5 rows
df_2017.head()

# Add 'Year' Column to the dfs
df_2017['Year'] = "2017-18"
df_2018['Year'] = "2018-19"
df_2019['Year'] = "2019-20"


# Append three dataframes into one
df = df_2017
df = df.append(df_2018)
df = df.append(df_2019)

df['Group'] = "CPD"
df.columns = df.columns.str.strip()
df.loc[((df['I would like to learn more about RHA'] == '`')),
       "I would like to learn more about RHA"] = 999
df = df.fillna(9999)
df = df.apply(lambda x: x.astype('int64') if x.dtype.kind in 'biufc' else x)
df['I would like to learn more about RHA'] = df['I would like to learn more about RHA'].astype(int)
# Exploring Variables

df.groupby(['Organization'])['ID number'].count()
df.groupby(['City'])['ID number'].count()
df.groupby(['Zip Code'])['ID number'].count()


# Renaming columns
# df = df.rename(columns = {'pre':'post'})

df = df.rename(columns={'I can identify a child having difficulty with asthma.1':
                        'I can identify a child having difficulty with asthma.2'})
df = df.rename(columns={'I can identify what can trigger an asthma attack.1':
                        'I can identify what can trigger an asthma attack.2'})
df = df.rename(columns={'I know what is needed to care for children with asthma.2':
                        'When using an asthma pump, it is best to take in the medication with a slow breath.2'})
df = df.rename(columns={'I know what is needed to care for children with asthma.1':
                        'I know what is needed to care for children with asthma.2'})
df = df.rename(columns={'Quick-relief inhalers should be used immediately .1':
                        'Quick-relief inhalers should be used immediately .2'})
df = df.rename(columns={'Long-term controller medicine should be taken ':
                        'Long-term controller medicine should be taken every day.2'})

df_cpd = df.copy()

#######
# Parents
#######

path = os.path.join(os.getcwd(), 'desktop', 'AM_csvs', 'Parents')

# df_2017 contains 2017-18, df_2018 contains 2018-19 data and so on
df_2017 = pd.read_csv(os.path.join(path, '2017-18.csv'))
df_2018 = pd.read_csv(os.path.join(path, '2018-19.csv'))
df_2019 = pd.read_csv(os.path.join(path, '2019-20.csv'))

# Add 'Year' Column to the dfs
df_2017['Year'] = "2017-18"
df_2018['Year'] = "2018-19"
df_2019['Year'] = "2019-20"

df_2018.groupby(['School'])['ID number'].count()

# Append three dataframes into one
df = df_2017
df = df.append(df_2018)
df = df.append(df_2019)

df['Group'] = "Parents"

df = df.fillna(9999)
df = df.apply(lambda x: x.astype('int64') if x.dtype.kind in 'biufc' else x)
df['Language'] = df['Language'].str.strip()
df['City'] = df['City'].str.strip()
df['District'] = df['District'].str.strip()
df['School'] = df['School'].str.strip()
df['relationship to child'] = df['relationship to child'].str.strip()
df['relationship to child'] = df['relationship to child'].str.lower()

# Renaming columns
# df = df.rename(columns = {'pre':'post'})

df = df.rename(columns={'Can Identify Child Difficulty w Asthma.1':
                        'Can Identify Child Difficulty w Asthma.2'})
df = df.rename(
    columns={'Identify Asthma Trigger.1':                          'Identify Asthma Trigger.2'})
df = df.rename(
    columns={'Know Care for Child w Asthma.1':                          'Know Care for Child w Asthma.2'})
df = df.rename(
    columns={'Slow breath with asthma pump.1':                          'Slow breath with asthma pump.2'})
df = df.rename(columns={
               'Quick Relief Used Immediately.1':                          'Quick Relief Used Immediately.2'})
df = df.rename(columns={
               'Long Term Controllers Used Daily.1':                          'Long Term Controllers Used Daily.2'})
df = df.rename(
    columns={'Carry and Self-Medicate Law.1':                          'Carry and Self-Medicate Law.2'})


df_parents = df.copy()


#######
# Childcare
#######

path = os.path.join(os.getcwd(), 'desktop', 'AM_csvs', 'Childcare')

df_2017 = pd.read_csv(os.path.join(path, '2017-18.csv'))
df_2018 = pd.read_csv(os.path.join(path, '2018-19.csv'))


# Add 'Year' Column to the dfs
df_2017['Year'] = "2017-18"
df_2018['Year'] = "2018-19"


# Append three dataframes into one
df = df_2017
df = df.append(df_2018)


df['Group'] = "Childcare"


df = df.fillna(9999)
df = df.apply(lambda x: x.astype('int64') if x.dtype.kind in 'biufc' else x)
df.columns = df.columns.str.strip()


# Exploring Variables

df.columns


# Renaming columns
# df = df.rename(columns = {'pre':'post'})

df = df.rename(columns={'I can identify a child having difficulty with asthma.1':
                        'I can identify a child having difficulty with asthma.2'})
df = df.rename(columns={'I can identify what can trigger an asthma attack..1':
                        'I can identify what can trigger an asthma attack.2'})
df = df.rename(columns={'I know what is needed to care for a child with asthma.1':
                        'I know what is needed to care for a child with asthma.2'})
df = df.rename(columns={'When using asthma pump, it is best to take medicine in with slow breath.1':
                        'When using asthma pump, it is best to take medicine in with slow breath.2'})
df = df.rename(columns={'quick relief inhalers used upon first warning sign.1':
                        'quick relief inhalers used upon first warning sign.2'})
df = df.rename(columns={'Long-term controller taken every day.1':
                        'Long-term controller taken every day.2'})
df = df.rename(columns={'I know how to use a nebulizer marchine.1':
                        'I know how to use a nebulizer marchine.2'})
df = df.rename(columns={'Unnamed: 30':                          'Notes'})

df_childcare = df.copy()


#######
# Staff (School)
#######

path = os.path.join(os.getcwd(), 'desktop', 'AM_csvs', 'Staff')

df_2017 = pd.read_csv(os.path.join(path, '2017-18.csv'))
df_2018 = pd.read_csv(os.path.join(path, '2018-19.csv'))
df_2019 = pd.read_csv(os.path.join(path, '2019-20.csv'))

# Add 'Year' Column to the dfs
df_2017['Year'] = "2017-18"
df_2018['Year'] = "2018-19"
df_2019['Year'] = "2019-20"


# Append three dataframes into one
df = df_2017
df = df.append(df_2018)
df = df.append(df_2019)


df['Group'] = "School Staff"


df['District'] = df['District'].str.strip()
df['City'] = df['City'].str.strip()
df['School'] = df['School'].str.strip()
df.loc[((df['Last Time Received Training'] == 'other')), "Last Time Received Training"] = 999
df.loc[((df['AAP on File at School'] == 'other')), "AAP on File at School"] = 999
df = df.fillna(9999)
df = df.apply(lambda x: x.astype('int64') if x.dtype.kind in 'biufc' else x)
df['AAP on File at School'] = df['AAP on File at School'].astype(int)
df['Last Time Received Training'] = df['Last Time Received Training'].astype(int)
# Renaming columns
# df = df.rename(columns = {'pre':'post'})

df = df.rename(columns={
               'Can Identify Child w Asthma':                          'Can Identify Child Difficulty w Asthma.2'})
df = df.rename(
    columns={'Identify Asthma Trigger.1':                          'Identify Asthma Trigger.2'})
df = df.rename(
    columns={'Know Care for Child w Asthma.1':                          'Know Care for Child w Asthma.2'})
df = df.rename(
    columns={'Slow breath with asthma pump.1':                          'Slow breath with asthma pump.2'})
df = df.rename(columns={
               'Quick Relief Used Immediately.1':                          'Quick Relief Used Immediately.2'})
df = df.rename(columns={
               'Long Term Controllers Used Daily.1':                          'Long Term Controllers Used Daily.2'})
df = df.rename(
    columns={'Carry and Self-Medicate Law.1':                          'Carry and Self-Medicate Law.2'})

df_staff = df.copy()


#######
# IDPH Staff
#######

path = os.path.join(os.getcwd(), 'desktop', 'AM_csvs', 'IDPH Staff')

df_2017 = pd.read_csv(os.path.join(path, '2017-18.csv'))

# Add 'Year' Column to the dfs
df_2017['Year'] = "2017-18"


df = df_2017


df['Group'] = "IDPH Staff"

df = df.fillna(9999)
df = df.apply(lambda x: x.astype('int64') if x.dtype.kind in 'biufc' else x)


# Exploring Variables
df.columns

# Renaming columns
# df = df.rename(columns = {'pre':'post'})

df = df.rename(columns={
               'Can Identify Child w Asthma':                          'Can Identify Child Difficulty w Asthma.2'})
df = df.rename(
    columns={'Identify Asthma Trigger.1':                          'Identify Asthma Trigger.2'})
df = df.rename(
    columns={'Know Care for Child w Asthma.1':                          'Know Care for Child w Asthma.2'})
df = df.rename(
    columns={'Slow breath with asthma pump.1':                          'Slow breath with asthma pump.2'})
df = df.rename(columns={
               'Quick Relief Used Immediately.1':                          'Quick Relief Used Immediately.2'})
df = df.rename(columns={
               'Long Term Controllers Used Daily.1':                          'Long Term Controllers Used Daily.2'})
df = df.rename(
    columns={'Carry and Self-Medicate Law.1':                          'Carry and Self-Medicate Law.2'})

df_idph = df.copy()


# Function to summarize responses for pre/post questions


def parents_summarize(df):
    test = pd.DataFrame(columns=["Questions",
                                 "Strongly Disagree",
                                 "Disagree",
                                 "Neutral",
                                 "Agree",
                                 "Strongly Agree",
                                 "Missing/Unknown"])

    questions = ['Can Identify Child Difficulty w Asthma',
                 'Identify Asthma Trigger',
                 'Know Care for Child w Asthma',
                 'Slow breath with asthma pump',
                 'Quick Relief Used Immediately',
                 'Long Term Controllers Used Daily',
                 'Carry and Self-Medicate Law',
                 'child carries quick-relief inhaler at school',
                 'child has AAP on file at school',
                 'Can Identify Child Difficulty w Asthma.2',
                 'Identify Asthma Trigger.2',
                 'Know Care for Child w Asthma.2',
                 'Slow breath with asthma pump.2',
                 'Quick Relief Used Immediately.2',
                 'Long Term Controllers Used Daily.2',
                 'Carry and Self-Medicate Law.2',
                 'Share information w/ others',
                 'Info increase asthma knowledge',
                 'presented in clear way',
                 'Recommend this program',
                 'will to participate in 3-month follow up',
                 'stock albuterol',
                 'Learn More/ RHA E-newsletter']

    for q in questions:
        r1 = len(df[df[q].isin({1})])  # Response 1, Strongly Disagree
        r2 = len(df[df[q].isin({2})])  # Response 2, Disagree
        r3 = len(df[df[q].isin({3})])  # Response 3, Neutral
        r4 = len(df[df[q].isin({4})])  # Response 4, Agree
        r5 = len(df[df[q].isin({5})])  # Response 5, Strongly Agree
        r6 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Strongly Disagree": r1,
                            "Disagree": r2,
                            "Neutral": r3,
                            "Agree": r4,
                            "Strongly Agree": r5,
                            "Missing/Unknown": r6}, ignore_index=True)

    return(test)


parents_summary = parents_summarize(df_parents)


df_parents.columns
df_parents.groupby(['barriers to submitting AAP '])['ID number'].count()

df_childcare.columns


def childcare_summarize(df):
    test = pd.DataFrame(columns=["Questions",
                                 "Strongly Disagree",
                                 "Disagree",
                                 "Neutral",
                                 "Agree",
                                 "Strongly Agree",
                                 "Missing/Unknown"])

    questions = ['I can identify a child having difficulty with asthma',
                 'I can identify what can trigger an asthma attack.',
                 'I know what is needed to care for a child with asthma',
                 'When using asthma pump, it is best to take medicine in with slow breath',
                 'quick relief inhalers used upon first warning sign',
                 'Long-term controller taken every day',
                 'I know how to use a nebulizer marchine',
                 'I know which students have an asthma action plan on file',
                 'last time I received asthma training',
                 'I can identify a child having difficulty with asthma.2',
                 'I can identify what can trigger an asthma attack.2',
                 'I know what is needed to care for a child with asthma.2',
                 'When using asthma pump, it is best to take medicine in with slow breath.2',
                 'quick relief inhalers used upon first warning sign.2',
                 'Long-term controller taken every day.2',
                 'I know how to use a nebulizer marchine.2',
                 'share info with others',
                 'instructor presented in clear way',
                 'recommend program to others',
                 'will participate in follow-up survey',
                 'favor requiring childcare staff to participate in asthma training at least',
                 'Learn More/ RHA E-newsletter']

    for q in questions:
        r1 = len(df[df[q].isin({1})])  # Response 1, Strongly Disagree
        r2 = len(df[df[q].isin({2})])  # Response 2, Disagree
        r3 = len(df[df[q].isin({3})])  # Response 3, Neutral
        r4 = len(df[df[q].isin({4})])  # Response 4, Agree
        r5 = len(df[df[q].isin({5})])  # Response 5, Strongly Agree
        r6 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Strongly Disagree": r1,
                            "Disagree": r2,
                            "Neutral": r3,
                            "Agree": r4,
                            "Strongly Agree": r5,
                            "Missing/Unknown": r6}, ignore_index=True)

    return(test)


childcare_summary = childcare_summarize(df_childcare)


df_staff.columns


def staff_summarize(df):
    test = pd.DataFrame(columns=["Questions",
                                 "Strongly Disagree",
                                 "Disagree",
                                 "Neutral",
                                 "Agree",
                                 "Strongly Agree",
                                 "Missing/Unknown"])

    questions = ['Can Identify Child Difficulty w Asthma', 'Identify Asthma Trigger',
                 'Know Care for Child w Asthma', 'Slow breath with asthma pump',
                 'Quick Relief Used Immediately', 'Long Term Controllers Used Daily',
                 'Carry and Self-Medicate Law', 'AAP on File at School',
                 'Last Time Received Training',
                 'Can Identify Child Difficulty w Asthma.2', 'Identify Asthma Trigger.2',
                 'Know Care for Child w Asthma.2', 'Slow breath with asthma pump.2',
                 'Quick Relief Used Immediately.2', 'Long Term Controllers Used Daily.2',
                 'Carry and Self-Medicate Law.2', 'Share information w/ others',
                 'Instructor Competence', 'Recommend this program', '3 month f/u survey',
                 'Stock Albuterol', 'Learn More/ RHA E-newsletter']

    for q in questions:
        r1 = len(df[df[q].isin({1})])  # Response 1, Strongly Disagree
        r2 = len(df[df[q].isin({2})])  # Response 2, Disagree
        r3 = len(df[df[q].isin({3})])  # Response 3, Neutral
        r4 = len(df[df[q].isin({4})])  # Response 4, Agree
        r5 = len(df[df[q].isin({5})])  # Response 5, Strongly Agree
        r6 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Strongly Disagree": r1,
                            "Disagree": r2,
                            "Neutral": r3,
                            "Agree": r4,
                            "Strongly Agree": r5,
                            "Missing/Unknown": r6}, ignore_index=True)

    return(test)


# df_staff.groupby(['Last Time Received Training'])['ID number'].count()
staff_summary = staff_summarize(df_staff)


df_idph.columns


def idph_summarize(df):
    test = pd.DataFrame(columns=["Questions",
                                 "Strongly Disagree",
                                 "Disagree",
                                 "Neutral",
                                 "Agree",
                                 "Strongly Agree",
                                 "Missing/Unknown"])

    questions = ['Can Identify Child Difficulty w Asthma',
                 'Identify Asthma Trigger',
                 'Know Care for Child w Asthma',
                 'Slow breath with asthma pump',
                 'Quick Relief Used Immediately',
                 'Long Term Controllers Used Daily',
                 'Carry and Self-Medicate Law',
                 'AAP on File at School',
                 'Last Time Received Training',
                 'Can Identify Child Difficulty w Asthma.2',
                 'Identify Asthma Trigger.2',
                 'Know Care for Child w Asthma.2',
                 'Slow breath with asthma pump.2',
                 'Quick Relief Used Immediately.2',
                 'Long Term Controllers Used Daily.2',
                 'Carry and Self-Medicate Law.2',
                 'Share information w/ others',
                 'Instructor Competence', 'Recommend this program',
                 '3 month f/u survey',
                 'Stock Albuterol']

    for q in questions:
        r1 = len(df[df[q].isin({1})])  # Response 1, Strongly Disagree
        r2 = len(df[df[q].isin({2})])  # Response 2, Disagree
        r3 = len(df[df[q].isin({3})])  # Response 3, Neutral
        r4 = len(df[df[q].isin({4})])  # Response 4, Agree
        r5 = len(df[df[q].isin({5})])  # Response 5, Strongly Agree
        r6 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Strongly Disagree": r1,
                            "Disagree": r2,
                            "Neutral": r3,
                            "Agree": r4,
                            "Strongly Agree": r5,
                            "Missing/Unknown": r6}, ignore_index=True)

    return(test)


idph_summary = idph_summarize(df_idph)


df_cpd.columns


def cpd_summarize(df):
    test = pd.DataFrame(columns=["Questions",
                                 "Strongly Disagree",
                                 "Disagree",
                                 "Neutral",
                                 "Agree",
                                 "Strongly Agree",
                                 "Missing/Unknown"])

    questions = ['I can identify a child having difficulty with asthma',
                 'I can identify what can trigger an asthma attack',
                 'I know what is needed to care for children with asthma',
                 'When using an asthma pump, it is best to take in the medicine with a slow breath',
                 'Quick-relief inhalers should be used immediately',
                 'Long-term controller medicine should be taken every day',
                 'The last ime I received an asthma training was',
                 'I can identify a child having difficulty with asthma.2',
                 'I can identify what can trigger an asthma attack.2',
                 'I know what is needed to care for children with asthma.2',
                 'When using an asthma pump, it is best to take in the medication with a slow breath.2',
                 'Quick-relief inhalers should be used immediately .2',
                 'Long-term controller medicine should be taken',
                 'I will share this information with other staff, families or adults',
                 'The instructor presented in a way that was clear',
                 'I would like to learn more about RHA']

    for q in questions:
        r1 = len(df[df[q].isin({1})])  # Response 1, Strongly Disagree
        r2 = len(df[df[q].isin({2})])  # Response 2, Disagree
        r3 = len(df[df[q].isin({3})])  # Response 3, Neutral
        r4 = len(df[df[q].isin({4})])  # Response 4, Agree
        r5 = len(df[df[q].isin({5})])  # Response 5, Strongly Agree
        r6 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Strongly Disagree": r1,
                            "Disagree": r2,
                            "Neutral": r3,
                            "Agree": r4,
                            "Strongly Agree": r5,
                            "Missing/Unknown": r6}, ignore_index=True)

    return(test)


cpd_summary = cpd_summarize(df_cpd)


# Functions to create table with pre and post scores

# CPD
def score_cpd(df):
    test = pd.DataFrame(columns=["Questions", "Correct", "Incorrect", "Missing"])

    binary_questions = ['When using an asthma pump, it is best to take in the medicine with a slow breath',
                        'Quick-relief inhalers should be used immediately',
                        'Long-term controller medicine should be taken every day',
                        'The last ime I received an asthma training was',
                        'When using an asthma pump, it is best to take in the medication with a slow breath.2',
                        'Quick-relief inhalers should be used immediately .2',
                        'Long-term controller medicine should be taken',
                        'I will share this information with other staff, families or adults',
                        'The instructor presented in a way that was clear',
                        'I would like to learn more about RHA']

    likert_questions = ['I can identify a child having difficulty with asthma',
                        'I can identify what can trigger an asthma attack',
                        'I know what is needed to care for children with asthma',
                        'I can identify a child having difficulty with asthma.2',
                        'I can identify what can trigger an asthma attack.2',
                        'I know what is needed to care for children with asthma.2']

    for q in likert_questions:
        r1 = len(df[df[q].isin({4, 5})])  # Correct response (4,5)
        r2 = len(df[df[q].isin({1, 2, 3})])  # Incorrect response (1,2,3)
        r3 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Correct": r1,
                            "Incorrect": r2,
                            "Missing": r3}, ignore_index=True)

    for q in binary_questions:
        r1 = len(df[df[q].isin({1})])  # Correct response (4,5)
        r2 = len(df[df[q].isin({2, 3})])  # Incorrect response (1,2,3)
        r3 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Correct": r1,
                            "Incorrect": r2,
                            "Missing": r3}, ignore_index=True)

    return(test)


cpd = score_cpd(df_cpd)


# Parents
def score_parents(df):
    test = pd.DataFrame(columns=["Questions",
                                 "Correct",
                                 "Incorrect",
                                 "Missing"])

    likert_questions = ['Can Identify Child Difficulty w Asthma',
                        'Identify Asthma Trigger',
                        'Know Care for Child w Asthma',
                        'Can Identify Child Difficulty w Asthma.2',
                        'Identify Asthma Trigger.2',
                        'Know Care for Child w Asthma.2']

    binary_questions = ['Slow breath with asthma pump',
                        'Quick Relief Used Immediately',
                        'Long Term Controllers Used Daily',
                        'Carry and Self-Medicate Law',
                        'child carries quick-relief inhaler at school',
                        'child has AAP on file at school',
                        'Slow breath with asthma pump.2',
                        'Quick Relief Used Immediately.2',
                        'Long Term Controllers Used Daily.2',
                        'Carry and Self-Medicate Law.2',
                        'Share information w/ others',
                        'Info increase asthma knowledge',
                        'presented in clear way',
                        'Recommend this program',
                        'will to participate in 3-month follow up',
                        'stock albuterol',
                        'Learn More/ RHA E-newsletter']

    for q in likert_questions:
        r1 = len(df[df[q].isin({4, 5})])  # Correct response (4,5)
        r2 = len(df[df[q].isin({1, 2, 3})])  # Incorrect response (1,2,3)
        r3 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Correct": r1,
                            "Incorrect": r2,
                            "Missing": r3}, ignore_index=True)

    for q in binary_questions:
        r1 = len(df[df[q].isin({1})])  # Correct response (4,5)
        r2 = len(df[df[q].isin({2, 3})])  # Incorrect response (1,2,3)
        r3 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Correct": r1,
                            "Incorrect": r2,
                            "Missing": r3}, ignore_index=True)

    return(test)


parents = score_parents(df_parents)

# Childcare


def score_childcare(df):
    test = pd.DataFrame(columns=["Questions", "Correct", "Incorrect", "Missing"])

    binary_questions = ['When using asthma pump, it is best to take medicine in with slow breath',
                        'quick relief inhalers used upon first warning sign',
                        'Long-term controller taken every day',
                        'I know how to use a nebulizer marchine',
                        'I know which students have an asthma action plan on file',
                        'last time I received asthma training',
                        'When using asthma pump, it is best to take medicine in with slow breath.2',
                        'quick relief inhalers used upon first warning sign.2',
                        'Long-term controller taken every day.2',
                        'I know how to use a nebulizer marchine.2',
                        'share info with others',
                        'instructor presented in clear way',
                        'recommend program to others',
                        'will participate in follow-up survey',
                        'favor requiring childcare staff to participate in asthma training at least',
                        'Learn More/ RHA E-newsletter']

    likert_questions = ['I can identify a child having difficulty with asthma',
                        'I can identify what can trigger an asthma attack.',
                        'I know what is needed to care for a child with asthma',
                        'I can identify a child having difficulty with asthma.2',
                        'I can identify what can trigger an asthma attack.2',
                        'I know what is needed to care for a child with asthma.2', ]

    for q in likert_questions:
        r1 = len(df[df[q].isin({4, 5})])  # Correct response (4,5)
        r2 = len(df[df[q].isin({1, 2, 3})])  # Incorrect response (1,2,3)
        r3 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Correct": r1,
                            "Incorrect": r2,
                            "Missing": r3}, ignore_index=True)

    for q in binary_questions:
        r1 = len(df[df[q].isin({1})])  # Correct response (4,5)
        r2 = len(df[df[q].isin({2, 3})])  # Incorrect response (1,2,3)
        r3 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Correct": r1,
                            "Incorrect": r2,
                            "Missing": r3}, ignore_index=True)

    return(test)


childcare = score_childcare(df_childcare)

# IDPH


def score_idph(df):
    test = pd.DataFrame(columns=["Questions",
                                 "Correct",
                                 "Incorrect",
                                 "Missing"])

    binary_questions = ['Slow breath with asthma pump',
                        'Quick Relief Used Immediately',
                        'Long Term Controllers Used Daily',
                        'Carry and Self-Medicate Law',
                        'AAP on File at School',
                        'Last Time Received Training',
                        'Slow breath with asthma pump.2',
                        'Quick Relief Used Immediately.2',
                        'Long Term Controllers Used Daily.2',
                        'Carry and Self-Medicate Law.2',
                        'Share information w/ others',
                        'Instructor Competence',
                        'Recommend this program',
                        '3 month f/u survey',
                        'Stock Albuterol']

    likert_questions = ['Can Identify Child Difficulty w Asthma',
                        'Identify Asthma Trigger',
                        'Know Care for Child w Asthma',
                        'Can Identify Child Difficulty w Asthma.2',
                        'Identify Asthma Trigger.2',
                        'Know Care for Child w Asthma.2']

    for q in likert_questions:
        r1 = len(df[df[q].isin({4, 5})])  # Correct response (4,5)
        r2 = len(df[df[q].isin({1, 2, 3})])  # Incorrect response (1,2,3)
        r3 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Correct": r1,
                            "Incorrect": r2,
                            "Missing": r3}, ignore_index=True)

    for q in binary_questions:
        r1 = len(df[df[q].isin({1})])  # Correct response (4,5)
        r2 = len(df[df[q].isin({2, 3})])  # Incorrect response (1,2,3)
        r3 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Correct": r1,
                            "Incorrect": r2,
                            "Missing": r3}, ignore_index=True)

    return(test)


idph = score_idph(df_idph)

# staff


def score_staff(df):
    test = pd.DataFrame(columns=["Questions",
                                 "Correct",
                                 "Incorrect",
                                 "Missing"])

    binary_questions = ['Slow breath with asthma pump',
                        'Quick Relief Used Immediately',
                        'Long Term Controllers Used Daily',
                        'Carry and Self-Medicate Law',
                        'AAP on File at School',
                        'Last Time Received Training',
                        'Slow breath with asthma pump.2',
                        'Quick Relief Used Immediately.2',
                        'Long Term Controllers Used Daily.2',
                        'Carry and Self-Medicate Law.2',
                        'Share information w/ others',
                        'Instructor Competence',
                        'Recommend this program',
                        '3 month f/u survey',
                        'Stock Albuterol',
                        'Learn More/ RHA E-newsletter']

    likert_questions = ['Can Identify Child Difficulty w Asthma',
                        'Identify Asthma Trigger',
                        'Know Care for Child w Asthma',
                        'Can Identify Child Difficulty w Asthma.2',
                        'Identify Asthma Trigger.2',
                        'Know Care for Child w Asthma.2']

    for q in likert_questions:
        r1 = len(df[df[q].isin({4, 5})])  # Correct response (4,5)
        r2 = len(df[df[q].isin({1, 2, 3})])  # Incorrect response (1,2,3)
        r3 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Correct": r1,
                            "Incorrect": r2,
                            "Missing": r3}, ignore_index=True)

    for q in binary_questions:
        r1 = len(df[df[q].isin({1})])  # Correct response (4,5)
        r2 = len(df[df[q].isin({2, 3})])  # Incorrect response (1,2,3)
        r3 = len(df[~df[q].isin({1, 2, 3, 4, 5})])    # Any other response
        test = test.append({"Questions": q,
                            "Correct": r1,
                            "Incorrect": r2,
                            "Missing": r3}, ignore_index=True)

    return(test)


staff = score_staff(df_staff)

# Parents barriers to allow carry

path = os.path.join(os.getcwd(), 'desktop', 'AM_csvs', 'Parents')

parents_q9 = df_parents['barriers to allow child to self-carry'].str.split(',', expand=True)
parents_q9 = parents_q9.apply(lambda x: x.str.strip())
parents_q9 = parents_q9.fillna(np.nan)
parents_q9['c0'] = parents_q9.eq('0').sum(axis=1)
parents_q9['c1'] = parents_q9.eq('1').sum(axis=1)
parents_q9['c2'] = parents_q9.eq('2').sum(axis=1)
parents_q9['c3'] = parents_q9.eq('3').sum(axis=1)
parents_q9['c4'] = parents_q9.eq('4').sum(axis=1)
parents_q9['c5'] = parents_q9.eq('5').sum(axis=1)
parents_q9['c6'] = parents_q9.eq('6').sum(axis=1)
parents_q9['c7'] = parents_q9.eq('7').sum(axis=1)
parents_q9['c8'] = parents_q9.eq('8').sum(axis=1)
parents_q9['c9'] = parents_q9.eq('9').sum(axis=1)
parents_q9['c10'] = parents_q9.eq('10').sum(axis=1)
parents_q9['c999'] = parents_q9.eq('999').sum(axis=1)
parents_q9.to_csv(os.path.join(path, 'parents_q9.csv'))


# Parents barriers to allow carry
# 'barriers to submitting AAP '

parents_q11 = df_parents['barriers to submitting AAP '].str.split(',', expand=True)
parents_q11 = parents_q11.apply(lambda x: x.str.strip())
parents_q11 = parents_q11.fillna(np.nan)
parents_q11['c0'] = parents_q11.eq('0').sum(axis=1)
parents_q11['c1'] = parents_q11.eq('1').sum(axis=1)
parents_q11['c2'] = parents_q11.eq('2').sum(axis=1)
parents_q11['c3'] = parents_q11.eq('3').sum(axis=1)
parents_q11['c4'] = parents_q11.eq('4').sum(axis=1)
parents_q11['c5'] = parents_q11.eq('5').sum(axis=1)
parents_q11['c6'] = parents_q11.eq('6').sum(axis=1)
parents_q11['c7'] = parents_q11.eq('7').sum(axis=1)
parents_q11['c8'] = parents_q11.eq('8').sum(axis=1)
parents_q11['c9'] = parents_q11.eq('9').sum(axis=1)
parents_q11['c10'] = parents_q11.eq('10').sum(axis=1)
parents_q11['c999'] = parents_q11.eq('999').sum(axis=1)
parents_q11.to_csv(os.path.join(path, 'parents_q11.csv'))


# Parents relationship to child
df_parents.groupby(['relationship to child'])['ID number'].count()


# Childcare position at school
df_childcare.groupby(['position at school'])['ID Number'].count()

# Childcare training: 'last time I received asthma training'
df_childcare.groupby(['last time I received asthma training'])['ID Number'].count()

# CPD training:  'The last ime I received an asthma training was'
df_cpd.groupby(['The last ime I received an asthma training was'])['ID number'].count()

# Staff training:  'Last Time Received Training'
df_staff.groupby(['Last Time Received Training'])['ID number'].count()

# Staff position at school:  'Position at School'
df_staff.groupby(['Position at School'])['ID number'].count()

# IDPH position at school:  'Position at School'
df_idph.groupby(['Position at School'])['ID number'].count()

# IDPH training: 'Last Time Received Training'
df_idph.groupby(['Last Time Received Training'])['ID number'].count()

# reclassifying into outcomes
# Yes/No/I don't know/Missing or Correct/Incorrect/I don't know/Missing


def outcomes(df, pre_questions, post_questions):

    df = df.fillna(9999)
    df = df.apply(lambda x: x.astype('int64') if x.dtype.kind in 'biufc' else x)
    cols = pre_questions + post_questions
    df[cols] = df[cols].astype(int)
    for col in cols:
        df[col] = df[col].apply(lambda x: int(x) if int(x) in [1, 2, 3, 999] else 999)

    test = pd.DataFrame(columns=["Questions", "Outcome 1", "Outcome 2", "Outcome 3", "Outcome 4"])

    for pre, post in zip(pre_questions, post_questions):
        o1 = len(df[(df[pre].isin({1})) & (df[post].isin({1}))])
        o3 = len(df[(df[pre].isin({2, 3, 999})) & (df[post].isin({1}))])
        o2 = len(df[(df[pre].isin({1})) & (df[post].isin({2, 3, 999}))])
        o4 = len(df[(df[pre].isin({2, 3, 999})) & (df[post].isin({2, 3, 999}))])
        test = test.append({"Questions": pre, "Outcome 1": o1, "Outcome 2": o2,
                            "Outcome 3": o3, "Outcome 4": o4}, ignore_index=True)

    return(test)


def outcomes_no_missing(df, pre_questions, post_questions):

    df = df.fillna(9999)
    df = df.apply(lambda x: x.astype('int64') if x.dtype.kind in 'biufc' else x)
    cols = pre_questions + post_questions
    df[cols] = df[cols].astype(int)
    for col in cols:
        df[col] = df[col].apply(lambda x: int(x) if int(x) in [1, 2, 3, 999] else 999)

    test = pd.DataFrame(columns=["Questions", "Outcome 1", "Outcome 2", "Outcome 3", "Outcome 4"])

    for pre, post in zip(pre_questions, post_questions):
        o1 = len(df[(df[pre].isin({1})) & (df[post].isin({1}))])
        o3 = len(df[(df[pre].isin({2, 3})) & (df[post].isin({1}))])
        o2 = len(df[(df[pre].isin({1})) & (df[post].isin({2, 3}))])
        o4 = len(df[(df[pre].isin({2, 3})) & (df[post].isin({2, 3}))])
        test = test.append({"Questions": pre, "Outcome 1": o1, "Outcome 2": o2,
                            "Outcome 3": o3, "Outcome 4": o4}, ignore_index=True)

    return(test)


# Parents
pre_list = ['Slow breath with asthma pump', 'Quick Relief Used Immediately',
            'Long Term Controllers Used Daily', 'Carry and Self-Medicate Law']
post_list = ['Slow breath with asthma pump.2', 'Quick Relief Used Immediately.2',
             'Long Term Controllers Used Daily.2', 'Carry and Self-Medicate Law.2']

parent_outcomes = outcomes(df_parents, pre_list, post_list)
parent_outcomes_nm = outcomes_no_missing(df_parents, pre_list, post_list)


# Childcare
pre_list = ['When using asthma pump, it is best to take medicine in with slow breath',
            'quick relief inhalers used upon first warning sign',
            'Long-term controller taken every day',
            'I know how to use a nebulizer marchine']

post_list = [x + str('.2') for x in pre_list]
childcare_outcomes = outcomes(df_childcare, pre_list, post_list)
childcare_outcomes_nm = outcomes_no_missing(df_childcare, pre_list, post_list)


# CPD
pre_list = ['When using an asthma pump, it is best to take in the medicine with a slow breath',
            'Quick-relief inhalers should be used immediately',
            'Long-term controller medicine should be taken every day']
post_list = ['When using an asthma pump, it is best to take in the medication with a slow breath.2',
             'Quick-relief inhalers should be used immediately .2',
             'Long-term controller medicine should be taken']

cpd_outcomes = outcomes(df_cpd, pre_list, post_list)
cpd_outcomes_nm = outcomes_no_missing(df_cpd, pre_list, post_list)


# staff
pre_list = ['Slow breath with asthma pump',
            'Quick Relief Used Immediately',
            'Long Term Controllers Used Daily',
            'Carry and Self-Medicate Law']
post_list = [x + str('.2') for x in pre_list]
staff_outcomes = outcomes(df_staff, pre_list, post_list)
staff_outcomes_nm = outcomes_no_missing(df_staff, pre_list, post_list)

# IDPH
pre_list = ['Slow breath with asthma pump', 'Quick Relief Used Immediately',
            'Long Term Controllers Used Daily', 'Carry and Self-Medicate Law']
post_list = [x + str('.2') for x in pre_list]
idph_outcomes = outcomes(df_idph, pre_list, post_list)
idph_outcomes_nm = outcomes_no_missing(df_idph, pre_list, post_list)


# Plots - Round 1
plt.rcParams['figure.dpi'] = 300
path = os.path.join(os.getcwd(), 'desktop', 'AM_csvs')
groups = {'Parents': parents_summary,
          'CPD': cpd_summary,
          'Staff': staff_summary,
          'IDPH': idph_summary,
          'Childcare Staff': childcare_summary}


# Likert scale questions

def likert(df, group):  # takes summary table as argument, returns pre/post % confident by counting Agree and Strongly Agree
    if group == 'CPD':
        table = pd.concat([df.iloc[:3, :], df.iloc[7:10, :]], axis=0)
    else:
        table = pd.concat([df.iloc[:3, :], df.iloc[9:12, :]], axis=0)
    total = table.iloc[:, 1:].sum(axis=1)[1]
    table['Confident'] = 100*table.iloc[:, 4:6].sum(axis=1)/total
    table = table[['Questions', 'Confident']]
    table = pd.concat([table.rename(columns={'Confident': 'Pre'}).iloc[:3, :],
                       table.rename(columns={'Confident': 'Post'}).iloc[3:, 1:].reset_index(drop=True)], axis=1)
    table = table.melt(id_vars=['Questions'], var_name='Type')
    table.drop('Questions', axis=1, inplace=True)
    q_col = list(parents_summary.iloc[:3, 0])
    q_col.extend(q_col)
    table['Questions'] = q_col
    return table


def plot_likert(table, group):
    sns.set()
    sns.set_style(style='ticks')
    colors = ["white", "grey"]
    fig, ax = plt.subplots(figsize=(12.5, 5))
#   plt.figure(figsize=(12.5,5))
    ax = sns.barplot(x="Questions", y='value', hue="Type",
                     palette=sns.xkcd_palette(colors),
                     edgecolor=(0, 0, 0),
                     linewidth=0.7,
                     data=table)

    ax.set_ylabel('Percent (%)', fontsize=18)
    ax.set_xlabel('Assessment Question', fontsize=18)
    ax.set_title(f'{group}: Change in Self-efficacy', pad=20, fontsize=20)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=16)
    ax.set_xticklabels(['Can identify symptoms',
                        'Can identify triggers',
                        'Knows how to care for asthma'], fontsize=14)
    sns.despine(top=True, right=True)
#   plt.legend(loc='upper center', ncol= 2, borderaxespad=0., frameon = False, fontsize = 14)
    plt.legend(bbox_to_anchor=(1, 0.6), fontsize=14, frameon=False,
               bbox_transform=plt.gcf().transFigure)

    plt.savefig(os.path.join(path, f'{group.lower()}-likert.jpg'), dpi=300)
    plt.show()
    plt.close()


def binary(df, group):  # takes summary table as argument, returns pre/post % confident by counting Agree and Strongly Agree
    if ((group == 'CPD') | (group == 'Childcare Staff')):
        if group == 'CPD':
            table = pd.concat([df.iloc[3:6, :], df.iloc[10:13, :]], axis=0).reset_index(drop=True)
        else:  # Childcare
            table = pd.concat([df.iloc[3:6, :], df.iloc[12:15, :]], axis=0).reset_index(drop=True)
        total = table.iloc[:, 1:].sum(axis=1)[0]
        table['Correct'] = 100*table.iloc[:, 1]/total
        table = table[['Questions', 'Correct']]
        table = pd.concat([table.rename(columns={'Correct': 'Pre'}).iloc[:3, :].reset_index(drop=True),
                           table.rename(columns={'Correct': 'Post'}).iloc[3:, 1:].reset_index(drop=True)], axis=1)
        table = table.melt(id_vars=['Questions'], var_name='Type')
        table.drop('Questions', axis=1, inplace=True)
        q_col = list(parents_summary.iloc[3:6, 0])
        q_col.extend(q_col)
        table['Questions'] = q_col
    else:  # Parents, IDPH, Staff
        table = pd.concat([df.iloc[3:7, :], df.iloc[12:16, :]], axis=0).reset_index(drop=True)
        total = table.iloc[:, 1:].sum(axis=1)[0]
        table['Correct'] = 100*table.iloc[:, 1]/total
        table = table[['Questions', 'Correct']]
        table = pd.concat([table.rename(columns={'Correct': 'Pre'}).iloc[:4, :].reset_index(drop=True),
                           table.rename(columns={'Correct': 'Post'}).iloc[4:, 1:].reset_index(drop=True)], axis=1)
        table = table.melt(id_vars=['Questions'], var_name='Type')
        table.drop('Questions', axis=1, inplace=True)
        q_col = list(parents_summary.iloc[3:7, 0])
        q_col.extend(q_col)
        table['Questions'] = q_col
    return table


def plot_binary(table, group):
    sns.set()
    sns.set_style(style='ticks')
    colors = ["white", "grey"]
    fig, ax = plt.subplots(figsize=(14, 5))
#   plt.figure(figsize=(12.5,5))
    ax = sns.barplot(x="Questions", y='value', hue="Type",
                     palette=sns.xkcd_palette(colors),
                     edgecolor=(0, 0, 0),
                     linewidth=0.7,
                     data=table)

    ax.set_ylabel('% Correct', fontsize=18)
    ax.set_xlabel('Assessment Question', fontsize=18)
    ax.set_title(f'{group}: Change in Knowledge', pad=20, fontsize=20)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=16)
    if ((group == 'CPD') | (group == 'Childcare Staff')):
        ax.set_xticklabels(['Slow breath w pump',
                            'Quick relief used immediately',
                            'Long term controllers daily'], fontsize=14)
    else:
        ax.set_xticklabels(['Slow breath w pump',
                            'Quick relief used immediately',
                            'Long term controllers daily',
                            'Items for inhaler carry and use'], fontsize=14)
    sns.despine(top=True, right=True)
#   plt.legend(loc='upper center', ncol= 2, borderaxespad=0., frameon = False, fontsize = 14)
    plt.legend(bbox_to_anchor=(1, 0.6), fontsize=14, frameon=False,
               bbox_transform=plt.gcf().transFigure)

    plt.savefig(os.path.join(path, f'{group.lower()}-binary.jpg'), dpi=300)
    plt.show()
    plt.close()


def plot_outcomes(table, group):
    ques = ['Slow breath w pump', 'Quick relief used immediately',
            'Long term controllers daily', 'Items for inhaler carry and use']
    # ques = ['Slow breath', 'Quick relief', 'Long term controllers', 'Self-medicate law']
    sns.set()
    sns.set_style(style='ticks')
    total = table.iloc[:, 1:].sum(axis=1)[1]
    table.iloc[:, 1:] = table.iloc[:, 1:]/(0.01*total)
#     colors = ["red","blue","green","pink"]

    fig, ax = plt.subplots(figsize=(12.5, 8))
    table = table.drop('Questions', axis=1)
    table = table.reset_index(drop=True)
    if group == 'CPD':
        table['Questions'] = ques[:-1]
    else:
        table['Questions'] = ques
    table = table.melt(id_vars=['Questions'], var_name='Outcomes')
    ax = sns.barplot(x="Questions", y='value', hue="Outcomes",
                     #                      palette = sns.xkcd_palette(colors),
                     edgecolor=(0, 0, 0),
                     linewidth=0.7,
                     data=table)

    ax.set_ylabel('Percent (%)', fontsize=18)
    ax.set_xlabel('Assessment Question', fontsize=18)
    ax.set_title(f'{group}: Outcomes', pad=20, fontsize=20)
#     plt.yticks(fontsize = 16)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=16)
    plt.xticks(fontsize=14)
    sns.despine(top=True, right=True)
    # plt.legend(loc='right', ncol= 1, borderaxespad=0., frameon = False, fontsize = 14)
    plt.legend(bbox_to_anchor=(1, 0.6), fontsize=14, frameon=False,
               bbox_transform=plt.gcf().transFigure)

    plt.savefig(os.path.join(path, f'{group.lower()}-outcomes.jpg'), dpi=300)
    plt.show()
    plt.close()


def plot_outcome3(table, group):
    ques = ['Slow breath w pump', 'Quick relief used immediately',
            'Long term controllers daily', 'Items for inhaler carry and use']
    # ques = ['Slow breath', 'Quick relief', 'Long term controllers', 'Self-medicate law']
    sns.set()
    sns.set_style(style='ticks')
    total = table.iloc[:, 1:].sum(axis=1)[1]
    table.iloc[:, 1:] = table.iloc[:, 1:]/(0.01*total)
    colors = ["white"]

    fig, ax = plt.subplots(figsize=(14, 5))
    table = table.drop('Questions', axis=1)
    table = table.reset_index(drop=True)
    if group == 'CPD':
        table['Questions'] = ques[:-1]
    else:
        table['Questions'] = ques
    ax = sns.barplot(x="Questions", y='Outcome 3',
                     palette=sns.xkcd_palette(colors),
                     edgecolor=(0, 0, 0),
                     linewidth=0.7,
                     data=table)

    ax.set_ylabel(f'% {group} with Improvement', fontsize=18)
    ax.set_xlabel('Assessment Question', fontsize=18)
    ax.set_title(f'{group}: Improvement in Knowledge', pad=20, fontsize=20)
#     plt.yticks(fontsize = 16)
    plt.yticks([0, 10, 20, 30, 40, 50], fontsize=16)
    plt.xticks(fontsize=14)
    sns.despine(top=True, right=True)
    # plt.legend(loc='right', ncol= 1, borderaxespad=0., frameon = False, fontsize = 14)

    plt.savefig(os.path.join(path, f'{group.lower()}-outcome3.jpg'), dpi=300)
    plt.show()
    plt.close()


path = os.path.join(os.getcwd(), 'desktop', 'AM_csvs')


groups = {'Parents': parents_summary,
          'CPD': cpd_summary,
          'Staff': staff_summary,
          'IDPH': idph_summary,
          'Childcare Staff': childcare_summary}

group_outcomes = {'Parents': parent_outcomes,
                  'CPD': cpd_outcomes,
                  'Staff': staff_outcomes,
                  'IDPH': idph_outcomes,
                  'Childcare Staff': childcare_outcomes}

group_outcomes_nm = {'Parents': parent_outcomes_nm,
                     'CPD': cpd_outcomes_nm,
                     'Staff': staff_outcomes_nm,
                     'IDPH': idph_outcomes_nm,
                     'Childcare Staff': childcare_outcomes_nm}


def mc_p_vals(table):
    # Getting McNemar's table as input
    p_vals = []
    for i in range(0, 3):  # 3 for CPD, 4 everywhere else
        matrix = [[table['Outcome 1'][i], table['Outcome 2'][i]],
                  [table['Outcome 3'][i], table['Outcome 4'][i]]]
        # calculate mcnemar test
        result = mcnemars_stat(matrix, exact=False, correction=True)
        p_vals.append(result.pvalue)

    p_vals = pd.DataFrame({'p-vals': p_vals})
    return p_vals

# mc_p_vals(staff_outcomes)


# Feb 01 - ppt changes

def outcomes_no_missing_2(df, pre_questions, post_questions):  # for likert scale questions

    df = df.fillna(9999)
    df = df.apply(lambda x: x.astype('int64') if x.dtype.kind in 'biufc' else x)
    cols = pre_questions + post_questions
    df[cols] = df[cols].astype(int)
    for col in cols:
        df[col] = df[col].apply(lambda x: int(x) if int(x) in [1, 2, 3, 4, 5, 999] else 999)

    test = pd.DataFrame(columns=["Questions", "Outcome 1", "Outcome 2", "Outcome 3", "Outcome 4"])

    for pre, post in zip(pre_questions, post_questions):
        o1 = len(df[(df[pre].isin({4, 5})) & (df[post].isin({4, 5}))])
        o3 = len(df[(df[pre].isin({1, 2, 3})) & (df[post].isin({4, 5}))])
        o2 = len(df[(df[pre].isin({4, 5})) & (df[post].isin({1, 2, 3}))])
        o4 = len(df[(df[pre].isin({1, 2, 3})) & (df[post].isin({1, 2, 3}))])
        test = test.append({"Questions": pre, "Outcome 1": o1, "Outcome 2": o2,
                            "Outcome 3": o3, "Outcome 4": o4}, ignore_index=True)

    return(test)


def plot_outcome3_2(table, group):  # for likert scale questions
    ques = ['Can identify symptoms', 'Can identify triggers', 'Knows how to care for asthma']
    sns.set()
    sns.set_style(style='ticks')
    total = table.iloc[:, 1:].sum(axis=1)[1]
    table.iloc[:, 1:] = table.iloc[:, 1:]/(0.01*total)
    colors = ["white"]

    fig, ax = plt.subplots(figsize=(14, 5))
    table = table.drop('Questions', axis=1)
    table = table.reset_index(drop=True)
    table['Questions'] = ques
    ax = sns.barplot(x="Questions", y='Outcome 3',
                     palette=sns.xkcd_palette(colors),
                     edgecolor=(0, 0, 0),
                     linewidth=0.7,
                     data=table)

    ax.set_ylabel(f'% {group} with Improvement', fontsize=18)
    ax.set_xlabel('Assessment Question', fontsize=18)
    ax.set_title(f'{group}: Improvement in Self-efficacy', pad=20, fontsize=20)
#     plt.yticks(fontsize = 16)
    plt.yticks([0, 10, 20, 30, 40, 50], fontsize=16)
    plt.xticks(fontsize=14)
    sns.despine(top=True, right=True)
    # plt.legend(loc='right', ncol= 1, borderaxespad=0., frameon = False, fontsize = 14)

    plt.savefig(os.path.join(path, f'{group.lower()}-outcome3_2.jpg'), dpi=300)
    plt.show()
    plt.close()


# Parents
pre_list_2 = ['Can Identify Child Difficulty w Asthma',
              'Identify Asthma Trigger',  'Know Care for Child w Asthma']
post_list_2 = ['Can Identify Child Difficulty w Asthma.2',
               'Identify Asthma Trigger.2', 'Know Care for Child w Asthma.2']

parent_outcomes_nm_2 = outcomes_no_missing_2(df_parents, pre_list_2, post_list_2)


# Childcare
pre_list_2 = ['I can identify a child having difficulty with asthma',
              'I can identify what can trigger an asthma attack.',
              'I know what is needed to care for a child with asthma']

post_list_2 = ['I can identify a child having difficulty with asthma.2',
               'I can identify what can trigger an asthma attack.2',
               'I know what is needed to care for a child with asthma.2']

childcare_outcomes_nm_2 = outcomes_no_missing_2(df_childcare, pre_list_2, post_list_2)


# CPD
pre_list_2 = ['I can identify a child having difficulty with asthma',
              'I can identify what can trigger an asthma attack',
              'I know what is needed to care for children with asthma']

post_list_2 = ['I can identify a child having difficulty with asthma.2',
               'I can identify what can trigger an asthma attack.2',
               'I know what is needed to care for children with asthma.2']

cpd_outcomes_nm_2 = outcomes_no_missing_2(df_cpd, pre_list_2, post_list_2)


# staff
pre_list_2 = ['Can Identify Child Difficulty w Asthma', 'Identify Asthma Trigger',
              'Know Care for Child w Asthma']

post_list_2 = ['Can Identify Child Difficulty w Asthma.2', 'Identify Asthma Trigger.2',
               'Know Care for Child w Asthma.2']

staff_outcomes_nm_2 = outcomes_no_missing_2(df_staff, pre_list_2, post_list_2)


# IDPH
pre_list_2 = ['Can Identify Child Difficulty w Asthma',
              'Identify Asthma Trigger', 'Know Care for Child w Asthma']
post_list_2 = [x + str('.2') for x in pre_list_2]


idph_outcomes_nm_2 = outcomes_no_missing_2(df_idph, pre_list_2, post_list_2)


group_outcomes_nm_2 = {'Parents': parent_outcomes_nm_2,
                       'CPD': cpd_outcomes_nm_2,
                       'Staff': staff_outcomes_nm_2,
                       'IDPH': idph_outcomes_nm_2,
                       'Childcare Staff': childcare_outcomes_nm_2}


# # self-efficacy
# mc_p_vals(parent_outcomes_nm_2)
# mc_p_vals(cpd_outcomes_nm_2)
# mc_p_vals(childcare_outcomes_nm_2)
# mc_p_vals(idph_outcomes_nm_2) # 0.077, 0, 0
# mc_p_vals(staff_outcomes_nm_2)

# # knowledge
# mc_p_vals(parent_outcomes_nm)
# mc_p_vals(cpd_outcomes_nm)
# mc_p_vals(childcare_outcomes_nm)
# mc_p_vals(idph_outcomes_nm)
# mc_p_vals(staff_outcomes_nm)


# Plotting

path = os.path.join(os.getcwd(), 'desktop', 'AM_csvs', 'testing')


for name, table in groups.items():
    plot_likert(likert(table, name), name)

for name, table in groups.items():
    plot_binary(binary(table, name), name)

# for name, table in group_outcomes.items():
#     plot_outcomes(table,name)

for name, table in group_outcomes.items():
    plot_outcome3(table, name)

# for name, table in group_outcomes_nm.items():
#     plot_outcomes(table,name)


for name, table in group_outcomes_nm_2.items():
    plot_outcome3_2(table, name)
