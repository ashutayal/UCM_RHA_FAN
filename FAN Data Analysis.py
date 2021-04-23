#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 3 2020
@author: ashutayal

"""
# Import libraries
from statsmodels.stats.contingency_tables import mcnemar as mcnemars_stat
from scipy.stats import anderson
from numpy.random import randn
from numpy.random import seed
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set working directory to location of csvs
# I put them in a folder called FAN_csvs on my desktop
path = os.path.join(os.getcwd(), 'desktop', 'FAN_csvs')

# df_2017 contains 2017-18, df_2018 contains 2018-19 data and so on
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

df.to_csv(os.path.join(path, 'df-raw.csv'))

df = df.fillna(9999)
df = df.apply(lambda x: x.astype('int64') if x.dtype.kind in 'biufc' else x)


# Removes white spaces at the beginning and end of strings
df['School Name'] = df['School Name'].str.strip()
df['District'] = df['District'].str.strip()
df['Type (Y/T)'] = df['Type (Y/T)'].str.strip()
df['Type (Y/T)'] = df['Type (Y/T)'].str.upper()


group_by_school = df.groupby(['School Name', 'Zip Code'])['ID #'].count()
group_by_school.to_frame()
# Group by schools, zip code

# School names, zip codes clean-up
df.loc[df['School Name'] == 'acero Esmeralda', ['School Name']] = 'Acero Esmeralda'
df.loc[df['School Name'] == 'joseph kellman corporate community school',
       ['School Name']] = 'Joseph Kellman Corporate Community School'
df.loc[df['School Name'] == 'Kellman Elementary School', [
    'School Name']] = 'Joseph Kellman Corporate Community School'
df.loc[df['School Name'] == 'okeefe elementary', ['School Name']] = "O'Keeffe School of Excellence"
df.loc[df['School Name'] == "O'Keefe Elementary", ['School Name']] = "O'Keeffe School of Excellence"
df.loc[df['School Name'] == "Wells Elementary", ['School Name']] = "Wells Elementary School"
df.loc[df['School Name'] == "William H Brown", [
    'School Name']] = "William H Brown Elementary School"
df.loc[df['School Name'] == "Wentworth Intermediate",
       ['School Name']] = "Wentworth Intermediate School"
df.loc[((df['School Name'] == "Wentworth Intermediate"))
       & ((df['Zip Code'] == 61081)), "Zip Code"] = 61801
df.loc[df['School Name'] == "Stagg Elementary", ['School Name']] = "Stagg Elementary School"
df.loc[df['School Name'] == "Spencer Elementary Technology Academy",
       ['School Name']] = "Spencer Technology Academy"
df.loc[df['School Name'] == "South Holland Kinder Care",
       ['School Name']] = "South Holland Kindercare"
df.loc[df['School Name'] == "Shabazz-Sizemore", ['School Name']] = "Betty Shabazz Sizemore"
df.loc[df['School Name'] == "Reavis Math and Science Specialty Elementary School",
       ['School Name']] = "Reavis Elementary School"
df.loc[df['School Name'] == "Posen Elementary", ['School Name']] = "Posen Elementary School"
df.loc[df['School Name'] == "Posen Elementary", ['School Name']] = "Posen Elementary School"
df.loc[df['School Name'] == "Posen Elementary", ['School Name']] = "Posen Elementary School"
df.loc[((df['School Name'] == "Plato Learning Academy Elementary School")) |
       ((df['School Name'] == "Plato Learning Academy Middle School")), "School Name"] = "Plato Learning Academy"
df.loc[df['School Name'] == "Posen Elementary", ['School Name']] = "Posen Elementary School"
df.loc[((df['School Name'] == "Perspectives Charter Middle School")) |
       ((df['School Name'] == "Perspectives Charter High School of Technology")), "School Name"] = "Perspectives Charter School"
df.loc[((df['School Name'] == "Paul Revere Elementary School")) |
       ((df['School Name'] == "Paul Revere Intermediate School")), "School Name"] = "Paul Revere School"
df.loc[df['School Name'] == "Parker", ['School Name']] = "Parker Elementary School"
df.loc[df['School Name'] == "Noble Charter Johnson Prep", [
    'School Name']] = "Noble Street - Johnson College Prep"
df.loc[df['School Name'] == "Noble Street-Rowe Clark",
       ['School Name']] = "Noble Street - Rowe Clark"
df.loc[df['School Name'] == "Nob Hill Elementary", ['School Name']] = "Nob Hill Elementary School"
df.loc[((df['School Name'] == "Nicholson Tech Elementary School"))
       & ((df['Zip Code'] == 60612)), "Zip Code"] = 60621
df.loc[df['School Name'] == "Nicholson Tech Elementary School",
       ['School Name']] = "Nicholson Tech Academy"
df.loc[df['School Name'] == "New Sullivan Elementary",
       ['School Name']] = "New Sullivan Elementary School"
df.loc[((df['School Name'] == "Nathan Hale Intermediate School")) |
       ((df['School Name'] == "Nathan Hale Middle School")), "School Name"] = "Nathan Hale School"
df.loc[df['School Name'] == "Mozart Elementary", ['School Name']] = "Mozart Elementary School"
df.loc[df['School Name'] == "McCall  Elementary", ['School Name']] = "McCall Elementary"
df.loc[df['School Name'] == "McAuliffe", ['School Name']] = "McAuliffe Elementary School"
df.loc[((df['School Name'] == "Lincoln Elementary")) & (
    (df['Zip Code'] == 60409)), "School Name"] = "Lincoln Elementary School"
df.loc[df['School Name'] == "Lewis Elementary School",
       ['School Name']] = "Leslie Lewis Elementary School"
df.loc[df['School Name'] == "Laura Ward Elementary",
       ['School Name']] = "Laura Ward Elementary School"
df.loc[df['School Name'] == "Kellogg Elementary", ['School Name']] = "Kellogg Elementary School"
df.loc[((df['School Name'] == "Kellar School")) & ((df['Zip Code'] == 60469)), "Zip Code"] = 60472
df.loc[df['School Name'] == "Johnnie Colemon Elementary",
       ['School Name']] = "Johnnie Colemon Elementary School"
df.loc[df['School Name'] == "Highlands Elementary", ['School Name']] = "Highlands Elementary School"
df.loc[df['School Name'] == "Hedges Elementary", ['School Name']] = "Hedges Elementary School"
df.loc[df['School Name'] == "Great Lakes Academy Charter School",
       ['School Name']] = "Great Lakes Academy"
df.loc[df['School Name'] == "George Manierre Elementary",
       ['School Name']] = "George Manierre Elementary School"
df.loc[df['School Name'] == "General George Patton", [
    'School Name']] = "General George Patton Elementary School"
df.loc[((df['School Name'] == "Frazier Prospective School"))
       & ((df['Zip Code'] == 4027)), "Zip Code"] = 60624
df.loc[((df['School Name'] == "Frazier Prospective IB Magnet School")) |
       ((df['School Name'] == "Frazier International Magnet Elementary School")), "School Name"] = "Frazier Prospective School"
df.loc[df['School Name'] == "Fieldcrest Elementary",
       ['School Name']] = "Fieldcrest Elementary School"
df.loc[df['School Name'] == "Evers Elementary", "Zip Code"] = 60628
df.loc[((df['School Name'] == "Dunne")) |
       ((df['School Name'] == "Dunne Tech Academy")), "School Name"] = "Dunne Technology Academy"
df.loc[((df['School Name'] == "East Dubuque High School")) |
       ((df['School Name'] == "East Dubuque Elementary")), "School Name"] = "East Dubuque School"
df.loc[df['School Name'] == "Earle Elementary", ['School Name']] = "Earle Elementary School"
df.loc[df['School Name'] == "Edward White Elementary School",
       ['School Name']] = "Edward White Career Academy"
df.loc[df['School Name'] == "Crown Elementary School", ['School Name']] = "Crown Community Academy"
df.loc[df['School Name'] == "Courtenay Elementary School",
       ['School Name']] = "Courtenay Language Arts Academy"
df.loc[df['School Name'] == "Chicago Heights Park District", "Zip Code"] = 60411
df.loc[df['School Name'] == "Carnegie Elementary", ['School Name']] = "Carnegie Elementary School"
df.loc[((df['School Name'] == "CICS- Longwood")) |
       ((df['School Name'] == "CICS-Longwood")), "School Name"] = "CICS - Longwood"
df.loc[df['School Name'] == "Carnegie Elementary", ['School Name']] = "Carnegie Elementary School"
df.loc[((df['School Name'] == "Brookwood Middle School")) |
       ((df['School Name'] == "Brookwood Junior High")), "School Name"] = "Brookwood School"
df.loc[df['School Name'] == "Brownell Elementary", ['School Name']] = "Brownell Elementary School"
df.loc[((df['School Name'] == "Betty Shabazz")) & (
    (df['Zip Code'] == 60619)), "School Name"] = "Betty Shabazz Academy"
df.loc[((df['School Name'] == "Betty Shabazz International")) & (
    (df['Zip Code'] == 60619)), "School Name"] = "Betty Shabazz Academy"
df.loc[((df['School Name'] == "Belmont Craign")) |
       ((df['School Name'] == "Belmont- Cragin Elementary School")), "School Name"] = "Belmont-Cragin Elementary School"
df.loc[df['School Name'] == "Belding", ['School Name']] = "Belding Elementary School"
df.loc[((df['School Name'] == "Alfred Nobel Elementary School"))
       & ((df['Zip Code'] == 60644)), "Zip Code"] = 60651
df.loc[((df['School Name'] == "George Washington Elementary School")) & (
    (df['Zip Code'] == 60803)), "School Name"] = "George Washington Elementary School - Alsip"
df.loc[((df['School Name'] == "George Washington Elementary School")) & (
    (df['Zip Code'] == 60617)), "School Name"] = "George Washington Elementary School - Chicago"
df.loc[((df['School Name'] == "Wentworth Elementary")) & ((df['Zip Code'] == 60409)),
       "School Name"] = "Wentworth Intermediate School - Calumet City"
df.loc[((df['School Name'] == "Wentworth Intermediate School")) & (
    (df['Zip Code'] == 60409)), "School Name"] = "Wentworth Intermediate School - Calumet City"
df.loc[((df['School Name'] == "wentworth Elementary")) & ((df['Zip Code'] == 60409)),
       "School Name"] = "Wentworth Intermediate School - Calumet City"
df.loc[df['School Name'] == "Pullman Elementary", [
    'School Name']] = "George Pullman Elementary School"
df.loc[df['School Name'] == "Lloyd Elementary School", "Zip Code"] = 60639
df.loc[df['School Name'] == "Parkside Elementary School", "Zip Code"] = 60649
df.loc[df['School Name'] == "Sabin Elementary School", "Zip Code"] = 60622
df.loc[df['School Name'] == "Brownell Elementary School", "Zip Code"] = 60637
df.loc[df['School Name'] == ".", "Zip Code"] = 999
df.loc[df['School Name'] == ".", "School Name"] = 999
df.loc[((df['School Name'] == "Paul Revere School")) & (
    (df['Zip Code'] == 60619)), "School Name"] = "Paul Revere School - Chicago"
df.loc[((df['School Name'] == "Paul Revere School")) & (
    (df['Zip Code'] == 60406)), "School Name"] = "Paul Revere School - Blue Island"
df.loc[((df['School Name'] == "Wentworth Elementary School")) & ((df['Zip Code'] == 60636)),
       "School Name"] = "Wentworth Elementary School - Chicago"
df.loc[df['School Name'] == "Urbana Middle School", "Zip Code"] = 61801
df.loc[df['School Name'] == "Emerson Elementary School", "Zip Code"] = 60402


# Race variable clean up
df.loc[df['Race'] == 12, "Race"] = 999
df.loc[df['Race'] == "1,2", "Race"] = 4
df.loc[df['Race'] == "1,3", "Race"] = 4
df.loc[df['Race'] == "2,3", "Race"] = 4
df.loc[df['Race'] == "2,3,5", "Race"] = 4
df.loc[df['Race'] == "2,3,5,6", "Race"] = 4
df.loc[df['Race'] == "2,4", "Race"] = 4
df.loc[df['Race'] == "2,5", "Race"] = 4
df.loc[df['Race'] == "2,5,6", "Race"] = 4
df.loc[df['Race'] == "2,7", "Race"] = 7
df.loc[df['Race'] == "2,6", "Race"] = 4
df.loc[df['Race'] == "3,5", "Race"] = 4
df.loc[df['Race'] == "3,6", "Race"] = 4
df.loc[df['Race'] == "3-Jan", "Race"] = 999
df.loc[df['Race'] == "4,5,6", "Race"] = 4
df.loc[df['Race'] == "4,7", "Race"] = 7
df.loc[df['Race'] == "5,6", "Race"] = 4
df.loc[df['Race'] == "5,7", "Race"] = 7
df.loc[df['Race'] == "No anser", "Race"] = 7
df.loc[df['Race'] == "no", "Race"] = 7

# Rename post columns to col.2
# df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
df = df.rename(columns={'1: Warning Signs': '1: Warning Signs.2',
                        '2: smoking trigger': '2: Smoking Trigger.2',
                        '3: Spacer.1': '3: Spacer.2',
                        '4: Slow Breath.1': '4: Slow Breath.2',
                        '5: Squeezing.1': '5: Squeezing.2',
                        '6:10-15.1': '6: 10-15.2',
                        '7:Swelling/Snot.1': '7: Swelling/Snot.2',
                        '8:Everyday.1': '8: Everyday.2',
                        '9:Calm': '9: Calm.2',
                        '10Exercise': '10: Exercise.2',
                        '11:Life-long.1': '11: Life-long.2',
                        '12:talk to adults.1': '12: Talk to Adults.2',
                        '13:talk AAP(post only)': '13: Talk AAP.2',
                        '14:inhaler at school': '14: Inhaler at School.2',
                        '15:comfortable carry': '15: Comfortable Carry.2',
                        '16:avoid triggers': '16: Avoid Triggers.2',
                        '17:episode at school': '17: Episode at School.2',
                        '18: I learned': '18: I Learned.2'})

# Rename pre columns to col.1
df = df.rename(columns={'1:  Warning Signs': '1: Warning Signs.1',
                        '2: Smoking Trigger': '2: Smoking Trigger.1',
                        '3: Spacer': '3: Spacer.1',
                        '4: Slow Breath': '4: Slow Breath.1',
                        '5: Squeezing': '5: Squeezing.1',
                        '6:10-15': '6: 10-15.1',
                        '7:Swelling/Snot': '7: Swelling/Snot.1',
                        '8:Everyday': '8: Everyday.1',
                        '9: Calm': '9: Calm.1',
                        '10:Exercise': '10: Exercise.1',
                        '11:Life-long': '11: Life-long.1',
                        '12:talk to adults': '12: Talk to Adults.1',
                        '13:inhaler at school': '13: Inhaler at School.1',
                        '14:comfortable carry': '14: Comfortable Carry.1',
                        '15:episode at school': '15: Episode at School.1'})


group_by_school_clean = df.groupby(['School Name', 'Zip Code'])['ID #'].count()
group_by_school_clean = group_by_school_clean.to_frame()

# # Drop students with entire pre/post survey missing
df = df[df['1: Warning Signs.1'] != '999']
df = df[df['1: Warning Signs.1'] != 999]
df = df[df['1: Warning Signs.2'] != '999']
df = df[df['1: Warning Signs.2'] != 999]

# Still has students who answer at-least one question on the pre & post

# Pre and post survey response clean-up

# [Yes, I don't know] to invalid response
# [No, I don't know] to I don't know
# Anything else to invalid response

# Pre responses
df.groupby(['1: Warning Signs.1'])['ID #'].count()

df.groupby(['2: Smoking Trigger.1'])['ID #'].count()
df.loc[((df['2: Smoking Trigger.1'] == "1,3")), "2: Smoking Trigger.1"] = 999
df.loc[((df['2: Smoking Trigger.1'] == 11)), "2: Smoking Trigger.1"] = 999


df.groupby(['3: Spacer.1'])['ID #'].count()
df.loc[((df['3: Spacer.1'] == 11)) | ((df['3: Spacer.1'] == 13)), "3: Spacer.1"] = 999
df.loc[((df['3: Spacer.1'] == "1,2")), "3: Spacer.1"] = 999
df.loc[((df['3: Spacer.1'] == "2,3")), "3: Spacer.1"] = 3

df.groupby(['4: Slow Breath.1'])['ID #'].count()
df.loc[((df['4: Slow Breath.1'] == 31)), "4: Slow Breath.1"] = 999

df.groupby(['5: Squeezing.1'])['ID #'].count()
df.loc[((df['5: Squeezing.1'] == 33)), "5: Squeezing.1"] = 999
df.loc[((df['5: Squeezing.1'] == '13')), "5: Squeezing.1"] = 999
df.loc[((df['5: Squeezing.1'] == '1 and 3')), "5: Squeezing.1"] = 999
df.loc[((df['5: Squeezing.1'] == '1,2')), "5: Squeezing.1"] = 999
df.loc[((df['5: Squeezing.1'] == '1,3')), "5: Squeezing.1"] = 999
df.loc[((df['5: Squeezing.1'] == '2,3')), "5: Squeezing.1"] = 3
df.loc[((df['5: Squeezing.1'] == 'huh? 3')), "5: Squeezing.1"] = 3

df.groupby(['6: 10-15.1'])['ID #'].count()
df.loc[((df['6: 10-15.1'] == 32)), "6: 10-15.1"] = 999
df.loc[((df['6: 10-15.1'] == 33)), "6: 10-15.1"] = 999
df.loc[((df['6: 10-15.1'] == 11)), "6: 10-15.1"] = 999

df.groupby(['7: Swelling/Snot.1'])['ID #'].count()
df.loc[((df['7: Swelling/Snot.1'] == 13)), "7: Swelling/Snot.1"] = 999
df.loc[((df['7: Swelling/Snot.1'] == '1,3')), "7: Swelling/Snot.1"] = 999
df.loc[((df['7: Swelling/Snot.1'] == '2,3')), "7: Swelling/Snot.1"] = 3

df.groupby(['8: Everyday.1'])['ID #'].count()
df.loc[((df['8: Everyday.1'] == 33)), "8: Everyday.1"] = 999

df.groupby(['9: Calm.1'])['ID #'].count()
df.loc[((df['9: Calm.1'] == '11')), "9: Calm.1"] = 999
df.loc[((df['9: Calm.1'] == '1,2')), "9: Calm.1"] = 999

df.groupby(['10: Exercise.1'])['ID #'].count()
df.loc[((df['10: Exercise.1'] == '1,2')), "10: Exercise.1"] = 999
df.loc[((df['10: Exercise.1'] == '1,3')), "10: Exercise.1"] = 999
df.loc[((df['10: Exercise.1'] == '23')), "10: Exercise.1"] = 999


df.groupby(['11: Life-long.1'])['ID #'].count()
df.loc[((df['11: Life-long.1'] == '1,2')), "11: Life-long.1"] = 999
df.loc[((df['11: Life-long.1'] == 12)), "11: Life-long.1"] = 999
df.loc[((df['11: Life-long.1'] == 33)), "11: Life-long.1"] = 999


df.groupby(['12: Talk to Adults.1'])['ID #'].count()
df.loc[((df['12: Talk to Adults.1'] == 11)), "12: Talk to Adults.1"] = 999
df.loc[((df['12: Talk to Adults.1'] == 13)), "12: Talk to Adults.1"] = 999
df.loc[((df['12: Talk to Adults.1'] == '1,2')), "12: Talk to Adults.1"] = 999
df.loc[((df['12: Talk to Adults.1'] == '1,3')), "12: Talk to Adults.1"] = 999
df.loc[((df['12: Talk to Adults.1'] == '31')), "12: Talk to Adults.1"] = 999
df.loc[((df['12: Talk to Adults.1'] == "I don't have asthma")), "12: Talk to Adults.1"] = 999
df.loc[((df['12: Talk to Adults.1'] == 'ƒß')), "12: Talk to Adults.1"] = 999

df.groupby(['13: Inhaler at School.1'])['ID #'].count()
df.loc[((df['13: Inhaler at School.1'] == '1,2')), "13: Inhaler at School.1"] = 999
df.loc[((df['13: Inhaler at School.1'] == "don't have 1  2")), "13: Inhaler at School.1"] = 999

df.groupby(['14: Comfortable Carry.1'])['ID #'].count()
df.loc[((df['14: Comfortable Carry.1'] == 11)), "14: Comfortable Carry.1"] = 999
df.loc[((df['14: Comfortable Carry.1'] == '4')), "14: Comfortable Carry.1"] = 999
df.loc[((df['14: Comfortable Carry.1'] == "don’t have 1   1")), "14: Comfortable Carry.1"] = 999

test = df.groupby(['15: Episode at School.1'])['ID #'].count()

# Post responses
df.groupby(['1: Warning Signs.2'])['ID #'].count()
df.loc[((df['1: Warning Signs.2'] == '1,2')), "1: Warning Signs.2"] = 999
df.loc[((df['1: Warning Signs.2'] == '11')), "1: Warning Signs.2"] = 999
df.loc[((df['1: Warning Signs.2'] == '99')), "1: Warning Signs.2"] = 999

df.groupby(['2: Smoking Trigger.2'])['ID #'].count()
df.loc[((df['2: Smoking Trigger.2'] == '1,2')), "2: Smoking Trigger.2"] = 999
df.loc[((df['2: Smoking Trigger.2'] == 14)), "2: Smoking Trigger.2"] = 999

df.groupby(['3: Spacer.2'])['ID #'].count()
df.loc[((df['3: Spacer.2'] == '11')), "3: Spacer.2"] = 999
df.loc[((df['3: Spacer.2'] == 11)), "3: Spacer.2"] = 999
df.loc[((df['3: Spacer.2'] == '1,2')), "3: Spacer.2"] = 999
df.loc[((df['3: Spacer.2'] == '1,3')), "3: Spacer.2"] = 999
df.loc[((df['3: Spacer.2'] == '`1')), "3: Spacer.2"] = 999

df.groupby(['4: Slow Breath.2'])['ID #'].count()
df.loc[((df['4: Slow Breath.2'] == '1,2')), "4: Slow Breath.2"] = 999
df.loc[((df['4: Slow Breath.2'] == '11')), "4: Slow Breath.2"] = 999
df.loc[((df['4: Slow Breath.2'] == '22')), "4: Slow Breath.2"] = 999
df.loc[((df['4: Slow Breath.2'] == '9')), "4: Slow Breath.2"] = 999

df.groupby(['5: Squeezing.2'])['ID #'].count()
df.loc[((df['5: Squeezing.2'] == '1,2')), "5: Squeezing.2"] = 999
df.loc[((df['5: Squeezing.2'] == 11)), "5: Squeezing.2"] = 999

df.groupby(['6: 10-15.2'])['ID #'].count()
df.loc[((df['6: 10-15.2'] == '1,2')), "6: 10-15.2"] = 999
df.loc[((df['6: 10-15.2'] == 11)), "6: 10-15.2"] = 999

df.groupby(['7: Swelling/Snot.2'])['ID #'].count()
df.loc[((df['7: Swelling/Snot.2'] == '1,2')), "7: Swelling/Snot.2"] = 999
df.loc[((df['7: Swelling/Snot.2'] == '1,3')), "7: Swelling/Snot.2"] = 999
df.loc[((df['7: Swelling/Snot.2'] == 11)), "7: Swelling/Snot.2"] = 999

df.groupby(['8: Everyday.2'])['ID #'].count()
df.loc[((df['8: Everyday.2'] == 11)), "8: Everyday.2"] = 999
df.loc[((df['8: Everyday.2'] == 23)), "8: Everyday.2"] = 999
df.loc[((df['8: Everyday.2'] == '1,2')), "8: Everyday.2"] = 999

df.groupby(['9: Calm.2'])['ID #'].count()
df.loc[((df['9: Calm.2'] == 11)), "9: Calm.2"] = 999
df.loc[((df['9: Calm.2'] == '1,2')), "9: Calm.2"] = 999
df.loc[((df['9: Calm.2'] == '9')), "9: Calm.2"] = 999

df.groupby(['10: Exercise.2'])['ID #'].count()
df.loc[((df['10: Exercise.2'] == 11)), "10: Exercise.2"] = 999
df.loc[((df['10: Exercise.2'] == '1 sometimes')), "10: Exercise.2"] = 999
df.loc[((df['10: Exercise.2'] == '1,2')), "10: Exercise.2"] = 999

df.groupby(['11: Life-long.2'])['ID #'].count()
df.loc[((df['11: Life-long.2'] == '11')), "11: Life-long.2"] = 999
df.loc[((df['11: Life-long.2'] == '1,2')), "11: Life-long.2"] = 999
df.loc[((df['11: Life-long.2'] == 11)), "11: Life-long.2"] = 999
df.loc[((df['11: Life-long.2'] == 13)), "11: Life-long.2"] = 999

df.groupby(['12: Talk to Adults.2'])['ID #'].count()
df.loc[((df['12: Talk to Adults.2'] == 12)), "12: Talk to Adults.2"] = 999
df.loc[((df['12: Talk to Adults.2'] == '1,2')), "12: Talk to Adults.2"] = 999
df.loc[((df['12: Talk to Adults.2'] == '2,3')), "12: Talk to Adults.2"] = 3
df.loc[((df['12: Talk to Adults.2'] == "3 I don't know what FAN is???")), "12: Talk to Adults.2"] = 3

df.groupby(['13: Talk AAP.2'])['ID #'].count()
df.loc[((df['13: Talk AAP.2'] == '1,2')), "13: Talk AAP.2"] = 999
df.loc[((df['13: Talk AAP.2'] == 'alr/did')), "13: Talk AAP.2"] = 999

df.groupby(['14: Inhaler at School.2'])['ID #'].count()
df.loc[((df['14: Inhaler at School.2'] == '1,2')), "14: Inhaler at School.2"] = 999
df.loc[((df['14: Inhaler at School.2'] == 99)), "14: Inhaler at School.2"] = 999
df.loc[((df['14: Inhaler at School.2'] == '1,3')), "14: Inhaler at School.2"] = 999
df.loc[((df['14: Inhaler at School.2'] == 'but sometime (in yes column) circled 2')),
       "14: Inhaler at School.2"] = 999
df.loc[((df['14: Inhaler at School.2'] == 'in nurses office 1')), "14: Inhaler at School.2"] = 999

df.groupby(['15: Comfortable Carry.2'])['ID #'].count()
df.loc[((df['15: Comfortable Carry.2'] == 99)), "15: Comfortable Carry.2"] = 999
df.loc[((df['15: Comfortable Carry.2'] == '1,2')), "15: Comfortable Carry.2"] = 999
df.loc[((df['15: Comfortable Carry.2'] == '1,3')), "15: Comfortable Carry.2"] = 999

df.groupby(['16: Avoid Triggers.2'])['ID #'].count()
df.loc[((df['16: Avoid Triggers.2'] == '1 t1')), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == '1,2')), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == '?')), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'calm down and take a breath in and out and I could not try to run. ')),
       "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'tell my teacher to call my mom to take me home')),
       "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == "stay calm at all moments. ")), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'stay calm and take my pump')), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'take my inhaler with spacer ')), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'stay calm take it out and take it ')),
       "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'tell someone')), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'stay calm ')), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'stay calm and find a teacher ')), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == "take my quick relief and it that doesn't work I should call an adult ")),
       "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'go to the main room')), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'get my asthma pump and be calm ')), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'stay calm and get my inhaler ')), "16: Avoid Triggers.2"] = 999
df.loc[((df['16: Avoid Triggers.2'] == 'tell the office ')), "16: Avoid Triggers.2"] = 999

# df.groupby(['17: Episode at School.2'])['ID #'].count()
# df.groupby(['18: I Learned.2'])['ID #'].count()

df['1: Warning Signs.1'] = df['1: Warning Signs.1'].astype(int)
df['2: Smoking Trigger.1'] = df['2: Smoking Trigger.1'].astype(int)
df['3: Spacer.1'] = df['3: Spacer.1'].astype(int)
df['4: Slow Breath.1'] = df['4: Slow Breath.1'].astype(int)
df['5: Squeezing.1'] = df['5: Squeezing.1'].astype(int)
df['6: 10-15.1'] = df['6: 10-15.1'].astype(int)
df['7: Swelling/Snot.1'] = df['7: Swelling/Snot.1'].astype(int)
df['8: Everyday.1'] = df['8: Everyday.1'].astype(int)
df['9: Calm.1'] = df['9: Calm.1'].astype(int)
df['10: Exercise.1'] = df['10: Exercise.1'].astype(int)
df['11: Life-long.1'] = df['11: Life-long.1'].astype(int)
df['12: Talk to Adults.1'] = df['12: Talk to Adults.1'].astype(int)
df['13: Inhaler at School.1'] = df['13: Inhaler at School.1'].astype(int)
df['14: Comfortable Carry.1'] = df['14: Comfortable Carry.1'].astype(int)


df['1: Warning Signs.2'] = df['1: Warning Signs.2'].astype(int)
df['2: Smoking Trigger.2'] = df['2: Smoking Trigger.2'].astype(int)
df['3: Spacer.2'] = df['3: Spacer.2'].astype(int)
df['4: Slow Breath.2'] = df['4: Slow Breath.2'].astype(int)
df['5: Squeezing.2'] = df['5: Squeezing.2'].astype(int)
df['6: 10-15.2'] = df['6: 10-15.2'].astype(int)
df['7: Swelling/Snot.2'] = df['7: Swelling/Snot.2'].astype(int)
df['8: Everyday.2'] = df['8: Everyday.2'].astype(int)
df['9: Calm.2'] = df['9: Calm.2'].astype(int)
df['10: Exercise.2'] = df['10: Exercise.2'].astype(int)
df['11: Life-long.2'] = df['11: Life-long.2'].astype(int)
df['12: Talk to Adults.2'] = df['12: Talk to Adults.2'].astype(int)
df['13: Talk AAP.2'] = df['13: Talk AAP.2'].astype(int)
df['14: Inhaler at School.2'] = df['14: Inhaler at School.2'].astype(int)
df['15: Comfortable Carry.2'] = df['15: Comfortable Carry.2'].astype(int)
df['16: Avoid Triggers.2'] = df['16: Avoid Triggers.2'].astype(int)


# Type conversions
df['Zip Code'] = df['Zip Code'].astype('Int64')
df['Gender'] = df['Gender'].astype('Int64')
df['Ever Taken Asthma Class'] = df['Ever Taken Asthma Class'].astype('Int64')
df['Ever Taken FAN'] = df['Ever Taken FAN'].astype('Int64')
df['Race'] = df['Race'].astype(int)
df['3 or 4 day program'] = df['3 or 4 day program'].astype(int)

# Gender clean-up
df.groupby(['Gender'])['ID #'].count()
df.loc[((df['Gender'] == 3)), "Gender"] = 999
df.loc[((df['Gender'] == 4)), "Gender"] = 999
df.loc[((df['Gender'] == 11)), "Gender"] = 999

# Prior Asthma class clean-up
df.loc[((df['Ever Taken FAN'] == 3)), "Ever Taken FAN"] = 999
df.loc[((df['Ever Taken Asthma Class'] == 3)), "Ever Taken Asthma Class"] = 999

df.loc[((df['School Name'] == 'Frohardt Elementary')) &
       ((df['ID #'] == '2887')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'Mitchell Elementary')) &
       ((df['ID #'] == '6')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'Cherry Valley Elementary School')) &
       ((df['ID #'] == '1472')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'Cherry Valley Elementary School')) &
       ((df['ID #'] == '1475')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'Greenwood Elementary School')) &
       ((df['ID #'] == '77')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'John Clark Elementary School')) &
       ((df['ID #'] == '65')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'John Clark Elementary School')) &
       ((df['ID #'] == '394')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'McCall Elementary')) &
       ((df['ID #'] == '550')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'Franklin Middle School')) &
       ((df['ID #'] == '54')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'Penniman Elementary School')) &
       ((df['ID #'] == '825')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'Evelyn Alexander Elementary')) &
       ((df['ID #'] == '800')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'Evelyn Alexander Elementary')) &
       ((df['ID #'] == '801')), "Ever Taken FAN"] = 999
df.loc[((df['School Name'] == 'Evelyn Alexander Elementary')) &
       ((df['ID #'] == '824')), "Ever Taken FAN"] = 999

# Attendance clean-up
df['Day 1'] = df['Day 1'].astype(int)
df.groupby(['Day 1'])['ID #'].count()
df.loc[((df['Day 1'] == 11)), "Day 1"] = 999

df['Day 2'] = df['Day 2'].astype(int)
df.groupby(['Day 2'])['ID #'].count()

df.groupby(['Day 3'])['ID #'].count()
df.loc[((df['Day 3'] == '1/2? (listed 2 times for class with different attendance numbers)')), "Day 3"] = 1
df['Day 3'] = df['Day 3'].astype(int)
df.loc[((df['Day 3'] == 3)), "Day 3"] = 999

df.groupby(['Day 4'])['ID #'].count()
df['Day 4'] = df['Day 4'].astype(int)


# Lincoln elementary 2018-19 changed to 4 day program
df.loc[((df['School Name'] == 'Lincoln Elementary School')) &
       ((df['Year'] == '2018-19')), "3 or 4 day program"] = 4
df.loc[((df['School Name'] == 'Ellington Elementary')) &
       ((df['Year'] == '2019-20')), "3 or 4 day program"] = 3
df.loc[((df['School Name'] == 'Laura Ward Elementary School')) &
       ((df['Year'] == '2019-20')), "3 or 4 day program"] = 2
df.loc[((df['School Name'] == 'BT Washington Elmentary')) &
       ((df['Year'] == '2019-20')), "3 or 4 day program"] = 3

# Shoesmith 2019-20 changed to 3 day program
df.loc[((df['School Name'] == 'Shoesmith')) &
       ((df['Year'] == '2019-20')), "3 or 4 day program"] = 3
# YCCS West 2017-18 changed to 2 day program
df.loc[((df['School Name'] == 'YCCS West Alternative School ')) &
       ((df['Year'] == '2017-18')), "3 or 4 day program"] = 2


# Henderson elementary 2018-19, IDs 2797:2822 changed to 4 day program
henderson_ids = ['2797', '2798', '2799', '2800', '2801', '2802', '2803', '2804', '2805',
                 '2818', '2819', '2820', '2821', '2822', '2823']
df.loc[(df['School Name'] == 'Henderson Elementary') &
       (df['ID #'].isin(henderson_ids)), "3 or 4 day program"] = 4


# Creating new attendance variables
df['Days Attended'] = ((df['Day 1'] == 1).astype(int) + (df['Day 2'] == 1).astype(int) +
                       (df['Day 3'] == 1).astype(int) + (df['Day 4'] == 1).astype(int))

df['Attendance %'] = 100*df['Days Attended']/df['3 or 4 day program']
df.loc[((df['3 or 4 day program'] == 9999)), "Attendance %"] = 999
# df.groupby(['Attendance %'])['ID #'].count()

# test_df = df[(df['Attendance %'] > 100) & (df['Attendance %'] < 400)]

# District clean-up
# test = df.groupby(['District'])['ID #'].count()
df['District'] = df['District'].str.strip()
df.loc[((df['District'] == '100')), "District"] = "District 100"
df.loc[((df['District'] == '133')), "District"] = "District 133"
df.loc[((df['District'] == '143.5')), "District"] = "District 143.5"
df.loc[((df['District'] == '144')), "District"] = "District 144"
df.loc[((df['District'] == '148')), "District"] = "District 148"
df.loc[((df['District'] == '170')), "District"] = "District 170"
df.loc[((df['District'] == '229')), "District"] = "District 50"
df.loc[((df['District'] == '130')), "District"] = "District 130"
df.loc[((df['District'] == '151')), "District"] = "District 151"
df.loc[((df['District'] == '152.5')), "District"] = "District 152.5"
df.loc[((df['District'] == '155')), "District"] = "District 155"
df.loc[((df['District'] == '201')), "District"] = "District 201"
df.loc[((df['District'] == '50')), "District"] = "District 50"
df.loc[((df['District'] == '93')), "District"] = "District 93"
df.loc[((df['District'] == 'S.Cook')), "District"] = "S. Cook"
df.loc[((df['District'] == 'Urbana SD')), "District"] = "Urbana School District"
df['District'] = df['District'].str.title()
df.loc[((df['District'] == 'cps')), "District"] = "CPS"

# Grade clean up
# combine K & pre-K to K/Pre-K
df.loc[((df['Grade'] == 'K ')), "Grade"] = 'K/Pre-K'
df.loc[((df['Grade'] == 'K')), "Grade"] = 'K/Pre-K'
df.loc[((df['Grade'] == 'pre-k')), "Grade"] = 'K/Pre-K'
df.loc[((df['Grade'] == 103)), "Grade"] = 999
df.loc[((df['Grade'] == '999')), "Grade"] = 999
df.loc[((df['Grade'] == '103')), "Grade"] = 999
df.loc[((df['Grade'] == '11/12/19')), "Grade"] = 999
df.groupby(['Grade'])['ID #'].count()


# Drop by age
# Age invalid if <5 or =25
df = df[df.Age != 1]
df = df[df.Age != 2]
df = df[df.Age != 3]
df = df[df.Age != 4]
df = df[df.Age != 25]


# Drop 1 day program attendees
df = df[df['3 or 4 day program'] != 1]


# Attendance
# adding a variable to indicate at least one day attended
df['Attended atleast 1 day'] = ((df['Day 1'] == 1) | (df['Day 2'] == 1) |
                                (df['Day 3'] == 1) | (df['Day 4'] == 1)).astype(int)


df.groupby(['Year', 'Attended atleast 1 day'])['ID #'].count()


# 2017-18 percentages in 3/4 day program
# 1/3, 2/3, 3/3 - counts (percentages) and so on


# df.groupby(['Year', 'Type (Y/T)', '3 or 4 day program'])['Day 1'].count()

# # Replace 0s, 9999s and 999s with numpy NAs
# # Doing this allows us to find means and stds

cols = ["3 or 4 day program", "Day 1", "Day 2", "Day 3", "Day 4", "Type (Y/T)", "Ever Taken Asthma Class",
        "Ever Taken FAN", "Gender", "Race", "Age", "Grade", "1: Warning Signs.1", "2: Smoking Trigger.1",
        "3: Spacer.1", "4: Slow Breath.1", "5: Squeezing.1", "6: 10-15.1", "7: Swelling/Snot.1",
        "8: Everyday.1", "9: Calm.1", "10: Exercise.1", "11: Life-long.1", "12: Talk to Adults.1",
        "13: Inhaler at School.1", "14: Comfortable Carry.1", "1: Warning Signs.2",
        "2: Smoking Trigger.2", "3: Spacer.2", "4: Slow Breath.2", "5: Squeezing.2", "6: 10-15.2", "7: Swelling/Snot.2", "8: Everyday.2", "9: Calm.2", "10: Exercise.2",
        "11: Life-long.2", "12: Talk to Adults.2", "13: Talk AAP.2", "14: Inhaler at School.2",
        "15: Comfortable Carry.2", "16: Avoid Triggers.2", "Attendance %"]

# df_2 used for means (nas are excluded from means)
df_2 = df.copy()
df_2[cols] = df_2[cols].replace({999: np.nan, 0: np.nan, 9999: np.nan, '999': np.nan})

# df_3 used for counts
# Here 0s, 999s, 9999s are treated alike
df_3 = df.copy()
df_3[cols] = df_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
df_3['Race'] = df_3['Race'].replace({7: 999})  # 7 is no answer


# Group by Race, Age, Gender, Grade, District, Prior Asthma education, Prior Fan education (Overall)
by_age = df_3.groupby(['Year', 'Type (Y/T)', 'Age'])['ID #'].count()
by_age.to_csv(os.path.join(path, 'age.csv'))
by_race = df_3.groupby(['Year', 'Type (Y/T)', 'Race'])['ID #'].count()
by_race.to_csv(os.path.join(path, 'race.csv'))
by_gender = df_3.groupby(['Year', 'Type (Y/T)', 'Gender'])['ID #'].count()
by_gender.to_csv(os.path.join(path, 'gender.csv'))
by_grade = df_3.groupby(['Year', 'Type (Y/T)', 'Grade'])['ID #'].count()
by_grade.to_csv(os.path.join(path, 'grade.csv'))
by_type = df_3.groupby(['Year', 'Type (Y/T)'])['ID #'].count()
by_type.to_csv(os.path.join(path, 'type.csv'))
by_prior_training = df_3.groupby(['Year', 'Type (Y/T)', 'Ever Taken Asthma Class'])['ID #'].count()
by_prior_training.to_csv(os.path.join(path, 'prior.csv'))
by_prior_FAN = df_3.groupby(['Year', 'Type (Y/T)', 'Ever Taken FAN'])['ID #'].count()
by_prior_FAN.to_csv(os.path.join(path, 'prior_FAN.csv'))
school_count = df_3.groupby(['Year', 'School Name'])['ID #'].count()
school_count.to_csv(os.path.join(path, 'school_count.csv'))
by_district = df_3.groupby(['Year', 'District'])['ID #'].count()
by_district.to_csv(os.path.join(path, 'district.csv'))
by_length = df_3.groupby(['Year', 'Type (Y/T)', '3 or 4 day program'])['ID #'].count()
by_length.to_csv(os.path.join(path, 'length.csv'))
by_attendance = df_2.groupby(['Year', 'Type (Y/T)', '3 or 4 day program'])['Attendance %'].mean()
by_attendance.to_csv(os.path.join(path, 'attendance.csv'))


# Export as csv
group_by_school_clean.to_csv(os.path.join(path, 'schools-clean.csv'))
df.to_csv(os.path.join(path, 'df-clean.csv'))


# Type missing to 999
df.loc[df['Type (Y/T)'] == ".", ['Type (Y/T)']] = 999


# [f(x) if condition else g(x) for x in sequence]
# ['yes' if v == 1 else 'no' if v == 2 else 'idle' for v in l]


questions = ["1: Warning Signs.1", "2: Smoking Trigger.1",
             "3: Spacer.1", "4: Slow Breath.1", "5: Squeezing.1", "6: 10-15.1", "7: Swelling/Snot.1",
             "8: Everyday.1", "9: Calm.1", "10: Exercise.1", "11: Life-long.1", "12: Talk to Adults.1",
             "13: Inhaler at School.1", "14: Comfortable Carry.1", "1: Warning Signs.2",
             "2: Smoking Trigger.2", "3: Spacer.2", "4: Slow Breath.2", "5: Squeezing.2", "6: 10-15.2", "7: Swelling/Snot.2", "8: Everyday.2", "9: Calm.2", "10: Exercise.2",
             "11: Life-long.2", "12: Talk to Adults.2", "13: Talk AAP.2", "14: Inhaler at School.2",
             "15: Comfortable Carry.2", "16: Avoid Triggers.2"]


knowledge_questions = ["1: Warning Signs.1", "2: Smoking Trigger.1",
                       "3: Spacer.1", "4: Slow Breath.1", "5: Squeezing.1", "6: 10-15.1", "7: Swelling/Snot.1",
                       "8: Everyday.1", "9: Calm.1", "10: Exercise.1", "11: Life-long.1", "1: Warning Signs.2",
                       "2: Smoking Trigger.2", "3: Spacer.2", "4: Slow Breath.2", "5: Squeezing.2",
                       "6: 10-15.2", "7: Swelling/Snot.2", "8: Everyday.2", "9: Calm.2", "10: Exercise.2",
                       "11: Life-long.2"]


pre_questions = ["1: Warning Signs.1",
                 "2: Smoking Trigger.1",
                 "3: Spacer.1",
                 "4: Slow Breath.1",
                 "5: Squeezing.1",
                 "6: 10-15.1",
                 "7: Swelling/Snot.1",
                 "8: Everyday.1",
                 "9: Calm.1",
                 "10: Exercise.1",
                 "11: Life-long.1",
                 "12: Talk to Adults.1",
                 "13: Inhaler at School.1",
                 "14: Comfortable Carry.1"]


post_questions = ["1: Warning Signs.2",
                  "2: Smoking Trigger.2",
                  "3: Spacer.2",
                  "4: Slow Breath.2",
                  "5: Squeezing.2",
                  "6: 10-15.2",
                  "7: Swelling/Snot.2",
                  "8: Everyday.2",
                  "9: Calm.2",
                  "10: Exercise.2",
                  "11: Life-long.2",
                  "13: Talk AAP.2",  # changed order to match pre with post
                  "14: Inhaler at School.2",
                  "15: Comfortable Carry.2"]


def outcomes(df):
    df_3 = df.copy()
    df_3[cols] = df_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
    test = pd.DataFrame(columns=["Questions", "OK", "Desired", "Undesired", "Excluded"])

    for pre, post in zip(pre_questions, post_questions):
        ques = pre[:-2]
        okay = len(df_3[(df_3[pre].isin({1})) & (df_3[post].isin({1}))])
        desired = len(df_3[(df_3[pre].isin({2, 3})) & (df_3[post].isin({1}))])
        undesired = len(df_3[(df_3[pre].isin({1, 2, 3})) & (df_3[post].isin({2, 3}))])
        excluded = len(df_3[(df_3[pre] == 999) | (df_3[post] == (999))])
        test = test.append({"Questions": ques, "OK": okay, "Desired": desired,
                            "Undesired": undesired, "Excluded": excluded}, ignore_index=True)

    # ques = '12: Talk to Adults'
    # okay = len(df_3[(df_3['12: Talk to Adults.1'].isin({1})) & (df_3['13: Talk AAP.2'].isin({1}))])
    # desired = len(df_3[(df_3['12: Talk to Adults.1'].isin({2, 3})) & (df_3['13: Talk AAP.2'].isin({1}))])
    # undesired = len(df_3[(df_3['12: Talk to Adults.1'].isin({1,2,3})) & (df_3['13: Talk AAP.2'].isin({2,3}))])
    # excluded = len(df_3[(df_3['12: Talk to Adults.1'] == 999) | (df_3['13: Talk AAP.2'] == (999))])
    # test = test.append({"Questions":ques,"OK":okay, "Desired": desired, "Undesired": undesired, "Excluded": excluded}, ignore_index= True)

    # ques = '13: Inhaler at School'
    # okay = len(df_3[(df_3['13: Inhaler at School.1'].isin({1})) & (df_3['14: Inhaler at School.2'].isin({1}))])
    # desired = len(df_3[(df_3['13: Inhaler at School.1'].isin({2, 3})) & (df_3['14: Inhaler at School.2'].isin({1}))])
    # undesired = len(df_3[(df_3['13: Inhaler at School.1'].isin({1,2,3})) & (df_3['14: Inhaler at School.2'].isin({2,3}))])
    # excluded = len(df_3[(df_3['13: Inhaler at School.1'] == 999) | (df_3['14: Inhaler at School.2'] == (999))])
    # test = test.append({"Questions":ques,"OK":okay, "Desired": desired, "Undesired": undesired, "Excluded": excluded}, ignore_index= True)

    # ques = '14: Comfortable Carry'
    # okay = len(df_3[(df_3['14: Comfortable Carry.1'].isin({1})) & (df_3['15: Comfortable Carry.2'].isin({1}))])
    # desired = len(df_3[(df_3['14: Comfortable Carry.1'].isin({2, 3})) & (df_3['15: Comfortable Carry.2'].isin({1}))])
    # undesired = len(df_3[(df_3['14: Comfortable Carry.1'].isin({1,2,3})) & (df_3['15: Comfortable Carry.2'].isin({2,3}))])
    # excluded = len(df_3[(df_3['14: Comfortable Carry.1'] == 999) | (df_3['15: Comfortable Carry.2'] == (999))])
    # test = test.append({"Questions":ques,"OK":okay, "Desired": desired, "Undesired": undesired, "Excluded": excluded}, ignore_index= True)

    return(test)


# Scraping 85 zip codes for Chicago
url = 'https://zipcode.org/city/IL/CHICAGO'
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:76.0) Gecko/20100101 Firefox/76.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'lxml')

'60637' in soup.text  # Checking to see if scrape successful, it was!

info = soup.find('body')

# The text of interest is all contained in div tags
# Within this div tag, there's a tags that contain required info

unparsed_rows = []
parsed_rows = []
unparsed_rows.append([val.text for val in info.find_all('div')])

flat_list = [item for sublist in unparsed_rows for item in sublist]
# converting this nested list to a flat list

for row in flat_list[99:105]:
    if row != []:
        parsed_rows.append(row)

test = []
for row in parsed_rows:
    test.append(row.splitlines())

flat_list2 = [item for sublist in test for item in sublist]

zips_scratch = []
for row in flat_list2:
    if ((row != '') & (row != '\xa0')):
        zips_scratch.append(row)

# Strip first 5 characters from each list item in zips
zips = []
for item in zips_scratch:
    zips.append(item.strip(" Zip Code"))

len(zips)  # 85 zip codes for Chicago
zips = list(set(zips))
zips_chicago = [int(i) for i in zips]


# Scraping zip codes for cook county
# This will include Chicago zip codes

url = 'http://ciclt.net/sn/clt/capitolimpact/gw_ziplist.aspx?ClientCode=capitolimpact&State=il&StName=Illinois&StFIPS=17&FIPS=17031'
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:76.0) Gecko/20100101 Firefox/76.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'lxml')

'60637' in soup.text  # Checking to see if scrape successful, it was!

info = soup.find('table')

# The text of interest is all contained in div tags
# Within this div tag, there's a tags that contain required info

unparsed_rows = []
parsed_rows = []
unparsed_rows.append([val.text for val in info.find_all('td')])

flat_list = [item for sublist in unparsed_rows for item in sublist]
# just converting this nested list to a flat list

test = []
for row in flat_list:
    test.append(row.splitlines())

flat_list2 = [item for sublist in test for item in sublist]

# row = flat_list2[24]
# row[0]

test2 = []
for item in flat_list2:
    if len(item) > 0:
        if item[0] == '6':
            test2.append(item)

test2 = list(set(test2))  # removes duplicates
len(test2)  # 182 zip codes
testing = pd.read_csv(os.path.join(path, 'cook_zips.csv'))
zips_cook = testing['zips'].tolist()
zips_only_cook = [i for i in zips_cook if i not in zips_chicago]

# Separating dfs based on zip codes
chicago = df_3[df_3['Zip Code'].isin(zips_chicago)].reset_index(drop=True)
cook = df_3[df_3['Zip Code'].isin(zips_only_cook)].reset_index(drop=True)
illinois = df_3[~df_3['Zip Code'].isin(zips_cook)].reset_index(drop=True)

# Separating dfs based on repeat/non repeat participant in FAN
repeat = df_3[df_3['Ever Taken FAN'].isin([1])].reset_index(drop=True)
non_repeat = df_3[~df_3['Ever Taken FAN'].isin([1])].reset_index(drop=True)

# Separating based on Youth/Teen program
teen = df_3[df_3['Type (Y/T)'] == 'T'].reset_index(drop=True)
youth = df_3[df_3['Type (Y/T)'] == 'Y'].reset_index(drop=True)


# Filling out the excel

# Overall
# attendance
df_2.groupby(['Year', '3 or 4 day program'])['Attendance %'].mean()

# Age
df_2.groupby(['Year'])['Age'].mean()

df_2[df_2['Year'] == '2017-18']['Age'].describe()

# Gender
df_3.groupby(['Gender'])['ID #'].count()

# Grade
grade_groups = df_3.groupby(['Year', 'Grade'])['ID #'].count()

# Race
race_groups = df_3.groupby(['Year', 'Race'])['ID #'].count()

# Prior FAN experience
df_3.groupby(['Year', 'Ever Taken FAN'])['ID #'].count()

# Prior asthma class (Yes to either)
df_3[(df_3['Ever Taken FAN'] == 1) | (df_3['Ever Taken Asthma Class'] == 1)].groupby('Year')['ID #'].count()

# No prior experience (No to both)
df_3[(df_3['Ever Taken FAN'] == 2) & (df_3['Ever Taken Asthma Class'] == 2)].groupby('Year')['ID #'].count()

# Missing/Unknown
df_3[(df_3['Ever Taken FAN'].isin({1})) & (
    df_3['Ever Taken Asthma Class'].isin({1}))].groupby('Year')['ID #'].count()
df_3[((df_3['Ever Taken FAN'].isin({1})) & (df_3['Ever Taken Asthma Class'].isin({2})))].groupby(
    'Year')['ID #'].count()
df_3[((df_3['Ever Taken FAN'].isin({1})) & (df_3['Ever Taken Asthma Class'].isin({999})))].groupby(
    'Year')['ID #'].count()
df_3[((df_3['Ever Taken FAN'].isin({2})) & (df_3['Ever Taken Asthma Class'].isin({1})))].groupby(
    'Year')['ID #'].count()
df_3[((df_3['Ever Taken FAN'].isin({2})) & (df_3['Ever Taken Asthma Class'].isin({2})))].groupby(
    'Year')['ID #'].count()
df_3[((df_3['Ever Taken FAN'].isin({2})) & (df_3['Ever Taken Asthma Class'].isin({999})))].groupby(
    'Year')['ID #'].count()
df_3[((df_3['Ever Taken FAN'].isin({999})) & (df_3['Ever Taken Asthma Class'].isin({1})))].groupby(
    'Year')['ID #'].count()
df_3[((df_3['Ever Taken FAN'].isin({999})) & (df_3['Ever Taken Asthma Class'].isin({2})))].groupby(
    'Year')['ID #'].count()
df_3[((df_3['Ever Taken FAN'].isin({999})) & (
    df_3['Ever Taken Asthma Class'].isin({999})))].groupby('Year')['ID #'].count()

# Length of program attended
df_3.groupby(['Year', '3 or 4 day program'])['ID #'].count()

# Number of schools, districts, zip codes
df_3.groupby('Year')['School Name'].nunique()
df_3.groupby('Year')['District'].nunique()
df_3.groupby('Year')['Zip Code'].nunique()


# Based on program type
# Attendance
df_2[df_2['Type (Y/T)'] == 'Y'].groupby(['Year', '3 or 4 day program'])['Attendance %'].mean()
df_2[df_2['Type (Y/T)'] == 'T'].groupby(['Year', '3 or 4 day program'])['Attendance %'].mean()

# Age
df_2[df_2['Type (Y/T)'] == 'Y'].groupby('Year')['Age'].describe()
df_2[df_2['Type (Y/T)'] == 'T'].groupby('Year')['Age'].describe()


# Gender
df_3[df_3['Type (Y/T)'] == 'Y'].groupby(['Year', 'Gender'])['ID #'].count()
df_3[df_3['Type (Y/T)'] == 'T'].groupby(['Year', 'Gender'])['ID #'].count()

# Grade
grade_groups = df_3[df_3['Type (Y/T)'] == 'Y'].groupby(['Year',
                                                        'Type (Y/T)', 'Grade'])['ID #'].count()
grade_groups = df_3[df_3['Type (Y/T)'] == 'T'].groupby(['Year',
                                                        'Type (Y/T)', 'Grade'])['ID #'].count()

# Race
race_groups = df_3[df_3['Type (Y/T)'] == 'Y'].groupby(['Year',
                                                       'Type (Y/T)', 'Race'])['ID #'].count()
race_groups = df_3[df_3['Type (Y/T)'] == 'T'].groupby(['Year',
                                                       'Type (Y/T)', 'Race'])['ID #'].count()

# Prior FAN experience
df_3[df_3['Type (Y/T)'] == 'Y'].groupby(['Year', 'Ever Taken FAN', 'Type (Y/T)'])['ID #'].count()
df_3[df_3['Type (Y/T)'] == 'T'].groupby(['Year', 'Ever Taken FAN', 'Type (Y/T)'])['ID #'].count()

# Prior asthma class (Yes to either)
df_3[(df_3['Ever Taken FAN'] == 1) | (df_3['Ever Taken Asthma Class'] == 1)
     ].groupby(['Year', 'Type (Y/T)'])['ID #'].count()

# No prior experience (No to both)
df_3[(df_3['Ever Taken FAN'] == 2) & (df_3['Ever Taken Asthma Class'] == 2)
     ].groupby(['Year', 'Type (Y/T)'])['ID #'].count()

# Missing/Unknown
# Subtract 'No' from the number below
df_3[(df_3['Ever Taken FAN'].isin({2, 999})) & (df_3['Ever Taken Asthma Class'].isin({2, 999})) &
     (df_3['Type (Y/T)'] == 'Y')].groupby('Year')['ID #'].count()
df_3[(df_3['Ever Taken FAN'].isin({2, 999})) & (df_3['Ever Taken Asthma Class'].isin({2, 999})) &
     (df_3['Type (Y/T)'] == 'T')].groupby('Year')['ID #'].count()


# Length of program attended
df_3.groupby(['Year', 'Type (Y/T)', '3 or 4 day program'])['ID #'].count()

# Number of schools, districts, zip codes
df_3[df_3['Type (Y/T)'] == 'Y'].groupby('Year')['School Name'].nunique()
df_3[df_3['Type (Y/T)'] == 'T'].groupby('Year')['School Name'].nunique()

df_3[df_3['Type (Y/T)'] == 'Y'].groupby('Year')['District'].nunique()
df_3[df_3['Type (Y/T)'] == 'T'].groupby('Year')['District'].nunique()

df_3[df_3['Type (Y/T)'] == 'Y'].groupby('Year')['Zip Code'].nunique()
df_3[df_3['Type (Y/T)'] == 'T'].groupby('Year')['Zip Code'].nunique()


# Based on prior participation

# Non Repeaters
# non_repeat_2 used for means (nas are excluded from means)
non_repeat_2 = non_repeat.copy()
non_repeat_2[cols] = non_repeat_2[cols].replace(
    {999: np.nan, 0: np.nan, 9999: np.nan, '999': np.nan})

# non_repeat_3 used for counts
# Here 0s, 999s, 9999s are treated alike
non_repeat_3 = non_repeat.copy()
non_repeat_3[cols] = non_repeat_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
non_repeat_3['Race'] = non_repeat_3['Race'].replace({7: 999})  # 7 is 'no answer'

# Age
non_repeat_2.groupby('Year')['Age'].describe()

# Gender
non_repeat_3.groupby(['Year', 'Gender'])['ID #'].count()

# Grade
grade_groups = non_repeat_3.groupby(['Year', 'Grade'])['ID #'].count()

# Race
non_repeat_3.groupby(['Year', 'Race'])['ID #'].count()

# Prior FAN experience
non_repeat_3.groupby(['Year', 'Ever Taken FAN'])['ID #'].count()

# Prior asthma class (Yes to either)
non_repeat_3[(non_repeat_3['Ever Taken FAN'] == 1) | (
    non_repeat_3['Ever Taken Asthma Class'] == 1)].groupby(['Year'])['ID #'].count()

# No prior experience (No to both)
non_repeat_3[(non_repeat_3['Ever Taken FAN'] == 2) & (
    non_repeat_3['Ever Taken Asthma Class'] == 2)].groupby(['Year'])['ID #'].count()

# Missing/Unknown
# Subtract 'No' from the number below
non_repeat_3[(non_repeat_3['Ever Taken FAN'].isin({2, 999})) &
             (non_repeat_3['Ever Taken Asthma Class'].isin({2, 999}))].groupby('Year')['ID #'].count()

# Length of program attended
non_repeat_3.groupby(['Year', '3 or 4 day program'])['ID #'].count()

# Attendance
non_repeat_2.groupby(['Year', '3 or 4 day program'])['Attendance %'].mean()

# Number of schools, districts, zip codes
non_repeat_3.groupby('Year')['School Name'].nunique()
non_repeat_3.groupby('Year')['District'].nunique()
non_repeat_3.groupby('Year')['Zip Code'].nunique()


# Repeaters
# repeat_2 used for means (nas are excluded from means)
repeat_2 = repeat.copy()
repeat_2[cols] = repeat_2[cols].replace({999: np.nan, 0: np.nan, 9999: np.nan, '999': np.nan})

# repeat_3 used for counts
# Here 0s, 999s, 9999s are treated alike
repeat_3 = repeat.copy()
repeat_3[cols] = repeat_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
repeat_3['Race'] = repeat_3['Race'].replace({7: 999})  # 7 is 'no answer'

# Age
repeat_2.groupby('Year')['Age'].describe()

# Gender
repeat_3.groupby(['Year', 'Gender'])['ID #'].count()

# Grade
grade_groups = repeat_3.groupby(['Year', 'Grade'])['ID #'].count()

# Race
repeat_3.groupby(['Year', 'Race'])['ID #'].count()

# Prior FAN experience
repeat_3.groupby(['Year', 'Ever Taken FAN'])['ID #'].count()

# Prior asthma class (Yes to either)
repeat_3[(repeat_3['Ever Taken FAN'] == 1) | (
    repeat_3['Ever Taken Asthma Class'] == 1)].groupby(['Year'])['ID #'].count()

# No prior experience (No to both)
repeat_3[(repeat_3['Ever Taken FAN'] == 2) & (
    repeat_3['Ever Taken Asthma Class'] == 2)].groupby(['Year'])['ID #'].count()

# Missing/Unknown
# Subtract 'No' from the number below
repeat_3[(repeat_3['Ever Taken FAN'].isin({2, 999})) & (repeat_3['Ever Taken Asthma Class'].isin({2, 999})) &
         (repeat_3['Type (Y/T)'] == 'Y')].groupby('Year')['ID #'].count()
repeat_3[(repeat_3['Ever Taken FAN'].isin({2, 999})) & (repeat_3['Ever Taken Asthma Class'].isin({2, 999})) &
         (repeat_3['Type (Y/T)'] == 'T')].groupby('Year')['ID #'].count()

# Length of program attended
repeat_3.groupby(['Year', '3 or 4 day program'])['ID #'].count()

# Attendance
repeat_2.groupby(['Year', '3 or 4 day program'])['Attendance %'].mean()

# Number of schools, districts, zip codes
repeat_3.groupby('Year')['School Name'].nunique()
repeat_3.groupby('Year')['District'].nunique()
repeat_3.groupby('Year')['Zip Code'].nunique()


# Based on region

# Chicago

# chicago_2 used for means (nas are excluded from means)
chicago_2 = chicago.copy()
chicago_2[cols] = chicago_2[cols].replace({999: np.nan, 0: np.nan, 9999: np.nan, '999': np.nan})

# chicago_3 used for counts
# Here 0s, 999s, 9999s are treated alike
chicago_3 = chicago.copy()
chicago_3[cols] = chicago_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
chicago_3['Race'] = chicago_3['Race'].replace({7: 999})  # 7 is 'no answer'

# Age
chicago_2.groupby('Year')['Age'].describe()

# Gender
chicago_3.groupby(['Year', 'Gender'])['ID #'].count()

# Grade
grade_groups = chicago_3.groupby(['Year', 'Grade'])['ID #'].count()

# Race
zzz = chicago_3.groupby(['Year', 'Race'])['ID #'].count()

# Prior FAN experience
chicago_3.groupby(['Year', 'Ever Taken FAN'])['ID #'].count()

# Prior asthma class (Yes to either)
chicago_3[(chicago_3['Ever Taken FAN'] == 1) | (
    chicago_3['Ever Taken Asthma Class'] == 1)].groupby(['Year'])['ID #'].count()

# No prior experience (No to both)
chicago_3[(chicago_3['Ever Taken FAN'] == 2) & (
    chicago_3['Ever Taken Asthma Class'] == 2)].groupby(['Year'])['ID #'].count()

# Missing/Unknown
# Subtract 'No' from the number below
chicago_3[(chicago_3['Ever Taken FAN'].isin({2, 999})) & (
    chicago_3['Ever Taken Asthma Class'].isin({2, 999}))].groupby('Year')['ID #'].count()

# Length of program attended
chicago_3.groupby(['Year', '3 or 4 day program'])['ID #'].count()

# Attendance
chicago_2.groupby(['Year', '3 or 4 day program'])['Attendance %'].mean()

# Number of schools, districts, zip codes
chicago_3.groupby('Year')['School Name'].nunique()
chicago_3.groupby('Year')['District'].nunique()
chicago_3.groupby('Year')['Zip Code'].nunique()


# Suburban Cook County

# cook_2 used for means (nas are excluded from means)
cook_2 = cook.copy()
cook_2[cols] = cook_2[cols].replace({999: np.nan, 0: np.nan, 9999: np.nan, '999': np.nan})

# cook_3 used for counts
# Here 0s, 999s, 9999s are treated alike
cook_3 = cook.copy()
cook_3[cols] = cook_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
cook_3['Race'] = cook_3['Race'].replace({7: 999})  # 7 is 'no answer'

# Age
cook_2.groupby('Year')['Age'].describe()

# Gender
cook_3.groupby(['Year', 'Gender'])['ID #'].count()

# Grade
grade_groups = cook_3.groupby(['Year', 'Grade'])['ID #'].count()

# Race
zzz = cook_3.groupby(['Year', 'Race'])['ID #'].count()

# Prior FAN experience
cook_3.groupby(['Year', 'Ever Taken FAN'])['ID #'].count()

# Prior asthma class (Yes to either)
cook_3[(cook_3['Ever Taken FAN'] == 1) | (
    cook_3['Ever Taken Asthma Class'] == 1)].groupby(['Year'])['ID #'].count()

# No prior experience (No to both)
cook_3[(cook_3['Ever Taken FAN'] == 2) & (
    cook_3['Ever Taken Asthma Class'] == 2)].groupby(['Year'])['ID #'].count()

# Missing/Unknown
# Subtract 'No' from the number below
cook_3[(cook_3['Ever Taken FAN'].isin({2, 999})) & (
    cook_3['Ever Taken Asthma Class'].isin({2, 999}))].groupby('Year')['ID #'].count()

# Length of program attended
cook_3.groupby(['Year', '3 or 4 day program'])['ID #'].count()

# Attendance
cook_2.groupby(['Year', '3 or 4 day program'])['Attendance %'].mean()

# Number of schools, districts, zip codes
cook_3.groupby('Year')['School Name'].nunique()
cook_3.groupby('Year')['District'].nunique()
cook_3.groupby('Year')['Zip Code'].nunique()


# Illinois

# illinois_2 used for means (nas are excluded from means)
illinois_2 = illinois.copy()
illinois_2[cols] = illinois_2[cols].replace({999: np.nan, 0: np.nan, 9999: np.nan, '999': np.nan})

# illinois_3 used for counts
# Here 0s, 999s, 9999s are treated alike
illinois_3 = illinois.copy()
illinois_3[cols] = illinois_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
illinois_3['Race'] = illinois_3['Race'].replace({7: 999})  # 7 is 'no answer'

# Age
illinois_2.groupby('Year')['Age'].describe()

# Gender
illinois_3.groupby(['Year', 'Gender'])['ID #'].count()

# Grade
grade_groups = illinois_3.groupby(['Year', 'Grade'])['ID #'].count()

# Race
zzz = illinois_3.groupby(['Year', 'Race'])['ID #'].count()

# Prior FAN experience
illinois_3.groupby(['Year', 'Ever Taken FAN'])['ID #'].count()

# Prior asthma class (Yes to either)
illinois_3[(illinois_3['Ever Taken FAN'] == 1) | (
    illinois_3['Ever Taken Asthma Class'] == 1)].groupby(['Year'])['ID #'].count()

# No prior experience (No to both)
illinois_3[(illinois_3['Ever Taken FAN'] == 2) & (
    illinois_3['Ever Taken Asthma Class'] == 2)].groupby(['Year'])['ID #'].count()

# Missing/Unknown
# Subtract 'No' from the number below
illinois_3[(illinois_3['Ever Taken FAN'].isin({2, 999})) & (
    illinois_3['Ever Taken Asthma Class'].isin({2, 999}))].groupby('Year')['ID #'].count()

# Length of program attended
illinois_3.groupby(['Year', '3 or 4 day program'])['ID #'].count()

# Attendance
illinois_2.groupby(['Year', '3 or 4 day program'])['Attendance %'].mean()

# Number of schools, districts, zip codes
illinois_3.groupby('Year')['School Name'].nunique()
illinois_3.groupby('Year')['District'].nunique()
illinois_3.groupby('Year')['Zip Code'].nunique()


# Defining some functions
# df_3 is a local variable (different from the global df_3)

def pivot_pre(df):
    df_3 = df.copy()
    df_3[cols] = df_3[cols].replace({3: 2, '3': 2})
    v1 = pd.pivot_table(df_3, index=['1: Warning Signs.1'], values='ID #', aggfunc=len)
    v2 = pd.pivot_table(df_3, index=['2: Smoking Trigger.1'], values='ID #', aggfunc=len)
    v3 = pd.pivot_table(df_3, index=['3: Spacer.1'], values='ID #', aggfunc=len)
    v4 = pd.pivot_table(df_3, index=['4: Slow Breath.1'], values='ID #', aggfunc=len)
    v5 = pd.pivot_table(df_3, index=['5: Squeezing.1'], values='ID #', aggfunc=len)
    v6 = pd.pivot_table(df_3, index=['6: 10-15.1'], values='ID #', aggfunc=len)
    v7 = pd.pivot_table(df_3, index=['7: Swelling/Snot.1'], values='ID #', aggfunc=len)
    v8 = pd.pivot_table(df_3, index=['8: Everyday.1'], values='ID #', aggfunc=len)
    v9 = pd.pivot_table(df_3, index=['9: Calm.1'], values='ID #', aggfunc=len)
    v10 = pd.pivot_table(df_3, index=['10: Exercise.1'], values='ID #', aggfunc=len)
    v11 = pd.pivot_table(df_3, index=['11: Life-long.1'], values='ID #', aggfunc=len)
    v12 = pd.pivot_table(df_3, index=['12: Talk to Adults.1'], values='ID #', aggfunc=len)
    v13 = pd.pivot_table(df_3, index=['13: Inhaler at School.1'], values='ID #', aggfunc=len)
    v14 = pd.pivot_table(df_3, index=['14: Comfortable Carry.1'], values='ID #', aggfunc=len)
    trial = pd.concat([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14], axis=1)
    trial = trial.transpose().reset_index(drop=True)
    return(trial)


def pivot_post(df):
    df_3 = df.copy()
    df_3[cols] = df_3[cols].replace({3: 2, '3': 2})
    v1 = pd.pivot_table(df_3, index=['1: Warning Signs.2'], values='ID #', aggfunc=len)
    v2 = pd.pivot_table(df_3, index=['2: Smoking Trigger.2'], values='ID #', aggfunc=len)
    v3 = pd.pivot_table(df_3, index=['3: Spacer.2'], values='ID #', aggfunc=len)
    v4 = pd.pivot_table(df_3, index=['4: Slow Breath.2'], values='ID #', aggfunc=len)
    v5 = pd.pivot_table(df_3, index=['5: Squeezing.2'], values='ID #', aggfunc=len)
    v6 = pd.pivot_table(df_3, index=['6: 10-15.2'], values='ID #', aggfunc=len)
    v7 = pd.pivot_table(df_3, index=['7: Swelling/Snot.2'], values='ID #', aggfunc=len)
    v8 = pd.pivot_table(df_3, index=['8: Everyday.2'], values='ID #', aggfunc=len)
    v9 = pd.pivot_table(df_3, index=['9: Calm.2'], values='ID #', aggfunc=len)
    v10 = pd.pivot_table(df_3, index=['10: Exercise.2'], values='ID #', aggfunc=len)
    v11 = pd.pivot_table(df_3, index=['11: Life-long.2'], values='ID #', aggfunc=len)
    v12 = pd.pivot_table(df_3, index=['12: Talk to Adults.2'], values='ID #', aggfunc=len)
    v13 = pd.pivot_table(df_3, index=['13: Talk AAP.2'], values='ID #', aggfunc=len)
    v14 = pd.pivot_table(df_3, index=['14: Inhaler at School.2'], values='ID #', aggfunc=len)
    v15 = pd.pivot_table(df_3, index=['15: Comfortable Carry.2'], values='ID #', aggfunc=len)
    v16 = pd.pivot_table(df_3, index=['16: Avoid Triggers.2'], values='ID #', aggfunc=len)
    trial = pd.concat([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                       v11, v16, v13, v14, v12, v15], axis=1)
    trial = trial.transpose().reset_index(drop=True)
    return(trial)


def undesired_outcomes(df):
    df_3 = df.copy()
    df_3[cols] = df_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
    test = pd.DataFrame(columns=["Questions", "C2In", "C2Id", "In2In", "In2Id", "Id2Id", "Id2In"])

    for pre, post in zip(pre_questions, post_questions):
        ques = pre[:-2]
        C2In = len(df_3[(df_3[pre].isin({1})) & (df_3[post].isin({2}))])
        C2Id = len(df_3[(df_3[pre].isin({1})) & (df_3[post].isin({3}))])
        In2In = len(df_3[(df_3[pre].isin({2})) & (df_3[post].isin({2}))])
        In2Id = len(df_3[(df_3[pre].isin({2})) & (df_3[post].isin({3}))])
        Id2Id = len(df_3[(df_3[pre].isin({3})) & (df_3[post].isin({3}))])
        Id2In = len(df_3[(df_3[pre].isin({3})) & (df_3[post].isin({2}))])
        test = test.append({"Questions": ques, "C2In": C2In, "C2Id": C2Id, "In2In": In2In,
                            "In2Id": In2Id, "Id2Id": Id2Id, "Id2In": Id2In}, ignore_index=True)

    return(test)


def mcnemar(df):
    df_3 = df.copy()
    df_3[cols] = df_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
    test = pd.DataFrame(columns=["Questions", "Outcome 1", "Outcome 2", "Outcome 3", "Outcome 4"])

    for pre, post in zip(pre_questions, post_questions):
        o1 = len(df[(df[pre].isin({1})) & (df[post].isin({1}))])
        o3 = len(df[(df[pre].isin({2, 3, 999})) & (df[post].isin({1}))])
        o2 = len(df[(df[pre].isin({1})) & (df[post].isin({2, 3, 999}))])
        o4 = len(df[(df[pre].isin({2, 3, 999})) & (df[post].isin({2, 3, 999}))])
        test = test.append({"Questions": pre[:-2], "Outcome 1": o1, "Outcome 2": o2,
                            "Outcome 3": o3, "Outcome 4": o4}, ignore_index=True)

    return(test)

# McNemar's test
# https://machinelearningmastery.com/mcnemars-test-for-machine-learning/


def mc_p_vals(df):
    # Getting McNemar's table
    table = mcnemar(df)
    table['Questions'] = table['Questions'].str[:3]
    table['Questions'] = table['Questions'].str.strip()
    table['Questions'] = table['Questions'].str.strip(':')
    p_vals = []
    # table['p-val'] = 999
    for i in range(0, 14):
        matrix = [[table['Outcome 1'][i], table['Outcome 2'][i]],
                  [table['Outcome 3'][i], table['Outcome 4'][i]]]
        # calculate mcnemar test
        result = mcnemars_stat(matrix, exact=False, correction=True)
        p_vals.append(result.pvalue)

    p_vals = pd.DataFrame({'p-vals': p_vals})
    return p_vals


# Final tables
overall_pre = pivot_pre(df_3)
overall_post = pivot_post(df_3)
overall_outcomes = outcomes(df_3)

repeat_pre = pivot_pre(repeat)
repeat_post = pivot_post(repeat)
repeat_outcomes = outcomes(repeat)

non_repeat_pre = pivot_pre(non_repeat)
non_repeat_post = pivot_post(non_repeat)
non_repeat_outcomes = outcomes(non_repeat)

teen_pre = pivot_pre(teen)
teen_post = pivot_post(teen)
teen_outcomes = outcomes(teen)

youth_pre = pivot_pre(youth)
youth_post = pivot_post(youth)
youth_outcomes = outcomes(youth)

chicago_pre = pivot_pre(chicago)
chicago_post = pivot_post(chicago)
chicago_outcomes = outcomes(chicago)

cook_pre = pivot_pre(cook)
cook_post = pivot_post(cook)
cook_outcomes = outcomes(cook)

illinois_pre = pivot_pre(illinois)
illinois_post = pivot_post(illinois)
illinois_outcomes = outcomes(illinois)

illinois_prior = illinois[illinois['Ever Taken FAN'] == 1].reset_index(drop=True)
illinois_prior.to_csv(os.path.join(path, 'illinois_prior.csv'))

overall_undesired = undesired_outcomes(df_3)


# Drop students with any answer missing (=999)

# for question in questions:
#     df_3 = df_3[-(df_3[question] == 999)]


# Creating a new column for pre score (out of 14), post score (out of 14),
# as well as knowledge pre and post scores (out of 11)


df_3['Pre Score'] = df_3[pre_questions].isin({1}).sum(1)  # sums along axis direction 1
df_3['Post Score'] = df_3[post_questions[0:11] + post_questions[12:15]].isin({1}).sum(1)
df_3['Pre Knowledge Score'] = df_3[pre_questions[0:11]].isin({1}).sum(1)
df_3['Post Knowledge Score'] = df_3[post_questions[0:11]].isin({1}).sum(1)


df_trial = overall_outcomes['Questions'].str[:3]
df_trial2 = pd.concat([df_trial.reset_index(drop=True), overall_pre[1]], axis=1)
df_trial2 = df_trial2.rename(columns={1: 'Pre'})
df_trial2 = pd.concat([df_trial2.reset_index(drop=True), overall_post.loc[np.r_[
                      0:11, 12:14, 15], 1].reset_index(drop=True)], axis=1)
df_trial2 = df_trial2.rename(columns={1: 'Post'})
df_trial2 = df_trial2.reset_index(drop=True)

df_trial2 = df_trial2.melt(id_vars=['Questions'], var_name='Type')
df_trial2['value'] = df_trial2['value']/34.5
df_trial2['Questions'] = df_trial2['Questions'].str.strip()
df_trial2['Questions'] = df_trial2['Questions'].str.strip(':')

# import itertools

# Plots

sns.set()

# sns.set_context(rc = {'patch.linewidth': 0.0})
sns.set_style(style='ticks')
colors = ["white", "grey", "medium green"]
plt.figure(figsize=(12.5, 5))
ax = sns.barplot(x="Questions", y='value', hue="Type", palette=sns.xkcd_palette(colors), edgecolor=(0, 0, 0), linewidth=0.7,
                 data=df_trial2, order=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"])

# plt.setp(ax.patches, linewidth = 0.5)
ax.set_ylabel('Percent of Students (%)', fontsize=18)
ax.set_xlabel('Assessment Question Number', fontsize=18)
ax.set_title(
    'Fight Asthma Now: Change in Knowledge, Practices, and Attitudes by Question', pad=20, fontsize=20)


# for i in range(11):
#     ax.get_xticklabels()[i].set_color("red")
# ax.get_xticklabels()[11].set_color("blue")
# ax.get_xticklabels()[12].set_color("blue")
# ax.get_xticklabels()[13].set_color("magenta")

# num_locations = 14
# hatches = itertools.cycle(['#', '/', '+', '+', 'x', '//', '*', 'o', 'O', '.'])
# for i, bar in enumerate(ax.patches):
#     if i % num_locations == 0:
#         hatch = next(hatches)
#     bar.set_hatch(hatch)


plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], fontsize=16)
plt.xticks(fontsize=16)
sns.despine(top=True, right=True)
plt.legend(loc='upper center', ncol=2, borderaxespad=0., frameon=False, fontsize=14)
# plt.subplots(figsize=(36,20))
plt.savefig(os.path.join(path, 'overall'), dpi=500)
plt.close()


# add Q1-Q11 - knowledge, Q12,13 - Practices, Q14 - Attitudes
# secondary x axis - label K/P/A

df_33 = df_3.copy()
for question in knowledge_questions:
    df_33 = df_33[-(df_33[question] == 999)]
len(df_33)

# pre assessment
sns.set()
sns.set_style(style='ticks')
ax = sns.barplot(x="Pre Knowledge Score", y="Pre Knowledge Score", data=df_33,  edgecolor=(0, 0, 0), linewidth=0.5,
                 estimator=lambda x: len(x) / len(df_33) * 100)
ax.set_ylabel('Percent of Students (%)', fontsize=13)
ax.set_xlabel('Score (0-11)', fontsize=13)
ax.set_title('Pre', pad=20, fontsize=16)
plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
plt.subplots_adjust(top=0.85)
sns.despine(top=True, right=True)
plt.savefig(os.path.join(path, 'pre'), dpi=300)  # Saves the plot to the repository folder
plt.close()

# post assessment
sns.set()
sns.set_style(style='ticks')
ax = sns.barplot(x="Post Knowledge Score", y="Post Knowledge Score", data=df_33,  edgecolor=(0, 0, 0), linewidth=0.5,
                 estimator=lambda x: len(x) / len(df_33) * 100)
# ax2 = ax.twinx()
# ax3 = sns.distplot(df_3["Post Knowledge Score"], ax=ax2, hist = False)
# ax3.tick_params(axis='both', which='both', length=0)
# ax3.yaxis.set_major_locator(plt.NullLocator())
ax.set_ylabel('Percent of Students (%)', fontsize=13)
ax.set_xlabel('Score (0-11)', fontsize=13)
# ax.set_xlabel('X_axi',fontsize=20);
ax.set_title('Post', pad=20, fontsize=16)
plt.subplots_adjust(top=0.85)
plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
sns.despine(top=True, right=True)
plt.savefig(os.path.join(path, 'post'), dpi=300)  # Saves the plot to the repository folder
plt.close()


# Separating dfs based on zip codes
chicago = df_3[df_3['Zip Code'].isin(zips_chicago)].reset_index(drop=True)
cook = df_3[df_3['Zip Code'].isin(zips_only_cook)].reset_index(drop=True)
illinois = df_3[~df_3['Zip Code'].isin(zips_cook)].reset_index(drop=True)


# Separating dfs based on repeat/non repeat participant in FAN
repeat = df_3[df_3['Ever Taken FAN'].isin([1])].reset_index(drop=True)
non_repeat = df_3[~df_3['Ever Taken FAN'].isin([1])].reset_index(drop=True)


# Separating based on Youth/Teen program
teen = df_3[df_3['Type (Y/T)'] == 'T'].reset_index(drop=True)
youth = df_3[df_3['Type (Y/T)'] == 'Y'].reset_index(drop=True)

# repeat['Post Knowledge Score'].mean() - repeat['Pre Knowledge Score'].mean()


# McNemar Outcomes

def mcnemar(df):  # no missing
    df_3 = df.copy()
    df_3[cols] = df_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
    test = pd.DataFrame(columns=["Questions", "Outcome 1", "Outcome 2", "Outcome 3", "Outcome 4"])

    for pre, post in zip(pre_questions, post_questions):
        o1 = len(df[(df[pre].isin({1})) & (df[post].isin({1}))])
        o3 = len(df[(df[pre].isin({2, 3})) & (df[post].isin({1}))])
        o2 = len(df[(df[pre].isin({1})) & (df[post].isin({2, 3}))])
        o4 = len(df[(df[pre].isin({2, 3})) & (df[post].isin({2, 3}))])
        test = test.append({"Questions": pre[:-2], "Outcome 1": o1, "Outcome 2": o2,
                            "Outcome 3": o3, "Outcome 4": o4}, ignore_index=True)

    return(test)


def mcnemar_v2(df):  # no missing
    df_3 = df.copy()
    df_3[cols] = df_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
    test = pd.DataFrame(columns=["Questions", "Outcome 1", "Outcome 2", "Outcome 3", "Outcome 4"])

    for pre, post in zip(pre_questions, post_questions):
        o1 = len(df[(df[pre].isin({1})) & (df[post].isin({1}))])
        o3 = len(df[(df[pre].isin({2, 3, 999})) & (df[post].isin({1}))])
        o2 = len(df[(df[pre].isin({1})) & (df[post].isin({2, 3, 999}))])
        o4 = len(df[(df[pre].isin({2, 3, 999})) & (df[post].isin({2, 3, 999}))])
        test = test.append({"Questions": pre[:-2], "Outcome 1": o1, "Outcome 2": o2,
                            "Outcome 3": o3, "Outcome 4": o4}, ignore_index=True)

    return(test)

# McNemar's test
# https://machinelearningmastery.com/mcnemars-test-for-machine-learning/


def mc_p_vals(df):
    # Getting McNemar's table
    table = mcnemar(df)
    table['Questions'] = table['Questions'].str[:3]
    table['Questions'] = table['Questions'].str.strip()
    table['Questions'] = table['Questions'].str.strip(':')
    p_vals = []
    # table['p-val'] = 999
    for i in range(0, 14):
        matrix = [[table['Outcome 1'][i], table['Outcome 2'][i]],
                  [table['Outcome 3'][i], table['Outcome 4'][i]]]
        # calculate mcnemar test
        result = mcnemars_stat(matrix, exact=False, correction=True)
        p_vals.append(result.pvalue)

    p_vals = pd.DataFrame({'p-vals': p_vals})
    return p_vals


mc_p_vals(non_repeat)


# plotting outcome 3
test = mcnemar(df_3)
test['Outcome 3'] = test['Outcome 3']*100/len(df_3)
test['Questions'] = test['Questions'].str[:3]
test['Questions'] = test['Questions'].str.strip()
test['Questions'] = test['Questions'].str.strip(':')

sns.set()
sns.set_style(style='ticks')
ax = sns.barplot(x="Questions", y="Outcome 3", data=test,  edgecolor=(0, 0, 0), linewidth=0.5,
                 order=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"])
ax.set_ylabel('Percent of Students (%)', fontsize=12)
ax.set_xlabel('Questions', fontsize=12)
ax.set_title('Fight Asthma Now: Improvement in Knowledge, Practices, and Attitudes', pad=20)
plt.subplots_adjust(top=0.85)
plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
sns.despine(top=True, right=True)
plt.savefig(os.path.join(path, 'mcnemar'), dpi=300)  # Saves the plot to the repository folder
plt.close()


# Creating a new df for region

region_scores = pd.DataFrame(columns=["Region", "Type", "Value"])
col1 = "Chicago"
col2 = "Pre"
col3 = chicago['Pre Knowledge Score'].mean()
region_scores = region_scores.append(
    {"Region": col1, "Type": col2, "Value": col3}, ignore_index=True)

col1 = "Chicago"
col2 = "Post"
col3 = chicago['Post Knowledge Score'].mean()
region_scores = region_scores.append(
    {"Region": col1, "Type": col2, "Value": col3}, ignore_index=True)

col1 = "Suburban Cook County"
col2 = "Pre"
col3 = cook['Pre Knowledge Score'].mean()
region_scores = region_scores.append(
    {"Region": col1, "Type": col2, "Value": col3}, ignore_index=True)

col1 = "Suburban Cook County"
col2 = "Post"
col3 = cook['Post Knowledge Score'].mean()
region_scores = region_scores.append(
    {"Region": col1, "Type": col2, "Value": col3}, ignore_index=True)

col1 = "Rest of IL"
col2 = "Pre"
col3 = illinois['Pre Knowledge Score'].mean()
region_scores = region_scores.append(
    {"Region": col1, "Type": col2, "Value": col3}, ignore_index=True)

col1 = "Rest of IL"
col2 = "Post"
col3 = illinois['Post Knowledge Score'].mean()
region_scores = region_scores.append(
    {"Region": col1, "Type": col2, "Value": col3}, ignore_index=True)


# Creating a new df for overall mean scores

overall_scores = pd.DataFrame(columns=["Type", "Value"])
col2 = "Pre"
col3 = df_3['Pre Knowledge Score'].mean()
overall_scores = overall_scores.append({"Type": col2, "Value": col3}, ignore_index=True)

col2 = "Post"
col3 = df_3['Post Knowledge Score'].mean()
overall_scores = overall_scores.append({"Type": col2, "Value": col3}, ignore_index=True)


# plotting t overall

sns.set()
sns.set_style(style='ticks')
colors = ["white", "white", "white"]
ax = sns.barplot(x="Type", y='Value', palette=sns.xkcd_palette(colors),
                 edgecolor=(0, 0, 0), linewidth=0.7,
                 data=overall_scores, order=["Pre", "Post"])

# plt.setp(ax.patches, linewidth = 0.5)
ax.set_ylabel('Mean Knowledge Score (out of 11)', fontsize=12)
ax.set_xlabel('Score Type', fontsize=12)
ax.set_title('Fight Asthma Now: Change in Knowledge', pad=20, fontsize=14)


plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.subplots_adjust(top=0.85)
sns.despine(top=True, right=True)
plt.legend(loc='upper center', ncol=2, borderaxespad=0., frameon=False)
plt.savefig(os.path.join(path, 't overall'), dpi=300)
plt.close()


# Plotting paired t-test

sns.set()
# sns.set_context(rc = {'patch.linewidth': 0.0})
sns.set_style(style='ticks')
colors = ["white", "grey"]
ax = sns.barplot(x="Region", y='Value', hue="Type", palette=sns.xkcd_palette(colors),
                 edgecolor=(0, 0, 0), linewidth=0.5,
                 data=region_scores, order=["Chicago", "Suburban Cook County", "Rest of IL"])

# plt.setp(ax.patches, linewidth = 0.5)
ax.set_ylabel('Mean Knowledge Score (out of 11)', fontsize=12)
ax.set_xlabel('Region', fontsize=12)
ax.set_title('Fight Asthma Now: Improvement in Knowledge', fontsize=14, pad=20)


# num_locations = 3
# hatches = itertools.cycle(['#', '/', '+', '+', 'x', '//', '*', 'o', 'O', '.'])
# for i, bar in enumerate(ax.patches):
#     if i % num_locations == 0:
#         hatch = next(hatches)
#     bar.set_hatch(hatch)


plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.subplots_adjust(top=0.85)
sns.despine(top=True, right=True)
plt.legend(loc='upper center', ncol=2, borderaxespad=0., frameon=False)
plt.savefig(os.path.join(path, 'paired-t region'), dpi=300)
plt.close()


# Plotting unpaired (2 sample) t-test

region_scores_pre = region_scores[region_scores['Type'] == "Pre"]

sns.set()
# sns.set_context(rc = {'patch.linewidth': 0.0})
sns.set_style(style='ticks')
colors = ["white", "white", "white"]
ax = sns.barplot(x="Region", y='Value', palette=sns.xkcd_palette(colors),
                 edgecolor=(0, 0, 0), linewidth=0.7,
                 data=region_scores_pre, order=["Chicago", "Suburban Cook County", "Rest of IL"])

# plt.setp(ax.patches, linewidth = 0.5)
ax.set_ylabel('Mean Baseline Knowledge Score (out of 11)', fontsize=12)
ax.set_xlabel('Region', fontsize=12)
ax.set_title('Fight Asthma Now: Comparison of Baseline Knowledge Scores', pad=20, fontsize=14)


plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.subplots_adjust(top=0.85)
sns.despine(top=True, right=True)
plt.legend(loc='upper center', ncol=2, borderaxespad=0., frameon=False)
plt.savefig(os.path.join(path, 'unpaired-t region'), dpi=300)
plt.close()


# Creating a new df for program type

type_scores = pd.DataFrame(columns=["Program", "Type", "Value"])
col1 = "Teen"
col2 = "Pre"
col3 = teen['Pre Knowledge Score'].mean()
type_scores = type_scores.append({"Program": col1, "Type": col2, "Value": col3}, ignore_index=True)

col1 = "Teen"
col2 = "Post"
col3 = teen['Post Knowledge Score'].mean()
type_scores = type_scores.append({"Program": col1, "Type": col2, "Value": col3}, ignore_index=True)

col1 = "Youth"
col2 = "Pre"
col3 = youth['Pre Knowledge Score'].mean()
type_scores = type_scores.append({"Program": col1, "Type": col2, "Value": col3}, ignore_index=True)

col1 = "Youth"
col2 = "Post"
col3 = youth['Post Knowledge Score'].mean()
type_scores = type_scores.append({"Program": col1, "Type": col2, "Value": col3}, ignore_index=True)


# Plotting paired t-test for program type

sns.set()
# sns.set_context(rc = {'patch.linewidth': 0.0})
sns.set_style(style='ticks')
colors = ["white", "grey"]
ax = sns.barplot(x="Program", y='Value', hue="Type", palette=sns.xkcd_palette(colors),
                 edgecolor=(0, 0, 0), linewidth=0.5,
                 data=type_scores, order=["Youth", "Teen"])

# plt.setp(ax.patches, linewidth = 0.5)
ax.set_ylabel('Mean Knowledge Score (out of 11)', fontsize=12)
ax.set_xlabel('Program Type', fontsize=12)
ax.set_title('Fight Asthma Now: Improvement in Knowledge', pad=20, fontsize=14)


# num_locations = 2
# hatches = itertools.cycle(['#', '/', '+', '+', 'x', '//', '*', 'o', 'O', '.'])
# for i, bar in enumerate(ax.patches):
#     if i % num_locations == 0:
#         hatch = next(hatches)
#     bar.set_hatch(hatch)

plt.subplots_adjust(top=0.85)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
sns.despine(top=True, right=True)
plt.legend(loc='upper center', ncol=2, borderaxespad=0., frameon=False)
plt.savefig(os.path.join(path, 'paired-t prog'), dpi=300)
plt.close()


# Plotting unpaired (2 sample) t-test

type_scores_pre = type_scores[region_scores['Type'] == "Pre"]

sns.set()
sns.set_style(style='ticks')
colors = ["white"]
ax = sns.barplot(x="Program", y='Value', palette=sns.xkcd_palette(colors), edgecolor=(0, 0, 0), linewidth=0.7,
                 data=type_scores_pre, order=["Youth", "Teen"])

# plt.setp(ax.patches, linewidth = 0.5)
ax.set_ylabel('Mean Baseline Knowledge Score (out of 11)', fontsize=12)
ax.set_xlabel('Program Type', fontsize=12)
ax.set_title('Fight Asthma Now: Comparison of Baseline Knowledge Scores', pad=20, fontsize=14)


plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.subplots_adjust(top=0.85)
sns.despine(top=True, right=True)
plt.legend(loc='upper center', ncol=2, borderaxespad=0., frameon=False)
plt.savefig(os.path.join(path, 'unpaired-t prog'), dpi=300)
plt.close()


# Creating a new df for prior FAN attendance/asthma education

fan_scores = pd.DataFrame(columns=["Prior FAN Experience", "Type", "Value"])
col1 = "Repeat Attendee"
col2 = "Pre"
col3 = repeat['Pre Knowledge Score'].mean()
fan_scores = fan_scores.append(
    {"Prior FAN Experience": col1, "Type": col2, "Value": col3}, ignore_index=True)

col1 = "Repeat Attendee"
col2 = "Post"
col3 = repeat['Post Knowledge Score'].mean()
fan_scores = fan_scores.append(
    {"Prior FAN Experience": col1, "Type": col2, "Value": col3}, ignore_index=True)

col1 = "Non-repeat Attendee"
col2 = "Pre"
col3 = non_repeat['Pre Knowledge Score'].mean()
fan_scores = fan_scores.append(
    {"Prior FAN Experience": col1, "Type": col2, "Value": col3}, ignore_index=True)

col1 = "Non-repeat Attendee"
col2 = "Post"
col3 = non_repeat['Post Knowledge Score'].mean()
fan_scores = fan_scores.append(
    {"Prior FAN Experience": col1, "Type": col2, "Value": col3}, ignore_index=True)


# Plotting paired t-test for program type

sns.set()
sns.set_style(style='ticks')
colors = ["white", "grey"]
ax = sns.barplot(x="Prior FAN Experience", y='Value', hue="Type",
                 palette=sns.xkcd_palette(colors), edgecolor=(0, 0, 0), linewidth=0.5,
                 data=fan_scores, order=["Non-repeat Attendee", "Repeat Attendee"])

# plt.setp(ax.patches, linewidth = 0.5)
ax.set_ylabel('Mean Knowledge Score (out of 11)', fontsize=12)
ax.set_xlabel('Prior FAN Experience', fontsize=12)
ax.set_title('Fight Asthma Now: Improvement in Knowledge', pad=20, fontsize=14)


# num_locations = 2
# hatches = itertools.cycle(['#', '/', '+', '+', 'x', '//', '*', 'o', 'O', '.'])
# for i, bar in enumerate(ax.patches):
#     if i % num_locations == 0:
#         hatch = next(hatches)
#     bar.set_hatch(hatch)

plt.subplots_adjust(top=0.85)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
ax.set_xticklabels(['First-time Attendee', 'Repeat Attendee'])
sns.despine(top=True, right=True)
plt.legend(loc='upper center', ncol=2, borderaxespad=0., frameon=False)
plt.savefig(os.path.join(path, 'paired-t fan'), dpi=300)
plt.close()


# Plotting unpaired (2 sample) t-test

fan_scores_pre = fan_scores[region_scores['Type'] == "Pre"]

sns.set()
sns.set_style(style='ticks')
colors = ["white"]
ax = sns.barplot(x="Prior FAN Experience", y='Value', palette=sns.xkcd_palette(colors), edgecolor=(0, 0, 0), linewidth=0.7,
                 data=fan_scores_pre, order=["Non-repeat Attendee", "Repeat Attendee"])

# plt.setp(ax.patches, linewidth = 0.5)
ax.set_ylabel('Mean Baseline Knowledge Score (out of 11)', fontsize=12)
ax.set_xlabel('Prior FAN Experience', fontsize=12)
ax.set_title('Fight Asthma Now: Comparison of Baseline Knowledge Scores', pad=20, fontsize=14)


plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
ax.set_xticklabels(['First-time Attendee', 'Repeat Attendee'])
plt.subplots_adjust(top=0.85)
sns.despine(top=True, right=True)
plt.legend(loc='upper center', ncol=2, borderaxespad=0., frameon=False)
plt.savefig(os.path.join(path, 'unpaired-t fan'), dpi=300)
plt.close()


# Statistical Testing

# Based on region


# Paired t tests, testing for normality using the Shapiro-Wilk test

# Overall

stats.wilcoxon(df_3['Pre Knowledge Score'], df_3['Post Knowledge Score'])
# p-val = 6.262864384682711e-267, highly significant

# Chicago

stats.shapiro(chicago['Pre Knowledge Score'])
# p-val = 1.5933553624605624e-20, normality rejected

stats.shapiro(chicago['Post Knowledge Score'])
# p-val = 0, normality rejected

# Can't use the paired-t test, use the Wilcoxon Sign-Ranked Test instead
stats.wilcoxon(chicago['Pre Knowledge Score'], chicago['Post Knowledge Score'])


# p-val = 3.322647256541453e-194, highly significant


# Cook

stats.shapiro(cook['Pre Knowledge Score'])
# p-val = 1.6952819148485787e-09, normality rejected

stats.shapiro(cook['Post Knowledge Score'])
# p-val = 6.7532733499938425e-31, normality rejected

# Can't use the paired-t test, use the Wilcoxon Sign-Ranked Test instead
stats.wilcoxon(cook['Pre Knowledge Score'], cook['Post Knowledge Score'])
# p-val = 5.286397678878269e-43, highly significant


# IL

stats.shapiro(illinois['Pre Knowledge Score'])
# p-val = 1.9567275577614396e-10, normality rejected

stats.shapiro(illinois['Post Knowledge Score'])
# p-val = 1.5015474292816197e-26, normality rejected

# Can't use the paired-t test, use the Wilcoxon Sign-Ranked Test instead
stats.wilcoxon(illinois['Pre Knowledge Score'], illinois['Post Knowledge Score'])
# p-val = pvalue=8.082169838384105e-36, highly significant


# Unpaired t-tests (2 sample t tests)

# Testing homogeneity of variances
# Levene's test: tests if variances are significantly different

# Chicago vs Cook
stats.levene(chicago['Pre Knowledge Score'], cook['Pre Knowledge Score'])
# P-val = 0.43459381087084714, not significant

# Chicago vs IL
stats.levene(chicago['Pre Knowledge Score'], illinois['Pre Knowledge Score'])
# P-val = 0.8166313782410539, not significant

# Cook vs IL
stats.levene(cook['Pre Knowledge Score'], illinois['Pre Knowledge Score'])
# P-val = 0.46802880295715954, not significant


# Normality is violated in all regions on both pre and post

# We use Mann-Whitney-Wilcoxon test instead of 2-sample t-test
# The Mann-Whitney U test is used when the assumptions of the independent samples t-test are violated.

# chicago vs cook
stats.mannwhitneyu(chicago['Pre Knowledge Score'], cook['Pre Knowledge Score'])
# p-val = 1.2552747491148342e-09, highly significant

# cook vs illinois
stats.mannwhitneyu(illinois['Pre Knowledge Score'], cook['Pre Knowledge Score'])
# p-val = 0.2251730745312649, not significant

# illinois vs chicago
stats.mannwhitneyu(illinois['Pre Knowledge Score'], chicago['Pre Knowledge Score'])
# p-val = 1.1672508589358612e-05, highly significant


# Based on program type
# Testing the normality: Shapiro-Wilk test

# Youth
stats.shapiro(youth['Pre Knowledge Score'])
# p-val = 1.679968786049233e-21, normality rejected

stats.shapiro(youth['Post Knowledge Score'])
# p-val = 0, normality rejected

# Paired test

# Can't use the paired-t test, use the Wilcoxon Sign-Ranked Test instead
stats.wilcoxon(youth['Pre Knowledge Score'], youth['Post Knowledge Score'])
# p-val = 3.440877976983705e-185, highly significant


# Teen
stats.shapiro(teen['Pre Knowledge Score'])
# p-val = 3.750957335534044e-13, normality rejected

stats.shapiro(teen['Post Knowledge Score'])
# p-val = 1.1171629200373776e-38, normality rejected

# Can't use the paired-t test, use the Wilcoxon Sign-Ranked Test instead
stats.wilcoxon(teen['Pre Knowledge Score'], teen['Post Knowledge Score'])
# p-val = 3.5331789942923655e-83, highly significant


# Normality violated in all cases

# Unpaired

# We use Mann-Whitney-Wilcoxon test instead of 2-sample t-test
# The Mann-Whitney U test is used when the assumptions of the independent samples t-test are violated.

# teen vs youth
stats.mannwhitneyu(teen['Pre Knowledge Score'], youth['Pre Knowledge Score'])
# p-val = 0.10010547264690295, insiginficant at 90% confidence level


# Based on prior FAN attendance
# Testing the normality: Shapiro-Wilk test

# First time
stats.shapiro(non_repeat['Pre Knowledge Score'])
# p-val = 7.171869597476756e-23, normality rejected

stats.shapiro(non_repeat['Post Knowledge Score'])
# p-val = 0, normality rejected

# Can't use the paired-t test, use the Wilcoxon Sign-Ranked Test instead
stats.wilcoxon(non_repeat['Pre Knowledge Score'], non_repeat['Post Knowledge Score'])
# p-val = 5.020170471083756e-223, highly significant


# Repeat
stats.shapiro(repeat['Pre Knowledge Score'])
# p-val = 1.2122546889925534e-12, normality rejected

stats.shapiro(repeat['Post Knowledge Score'])
# p-val = 6.135907171793343e-35, normality rejected

# Can't use the paired-t test, use the Wilcoxon Sign-Ranked Test instead
stats.wilcoxon(repeat['Pre Knowledge Score'], repeat['Post Knowledge Score'])
# p-val = 2.0374921968854784e-47, highly significant


# Normality violated in all cases

# Unpaired

# We use Mann-Whitney-Wilcoxon test instead of 2-sample t-test
# The Mann-Whitney U test is used when the assumptions of the independent samples t-test are violated.

# Repeat vs Non-repeat
stats.mannwhitneyu(repeat['Pre Knowledge Score'], non_repeat['Pre Knowledge Score'])
# p-val = 6.254939369478378e-71, highly significant


# # Trying out new graph

# from matplotlib import pyplot as plt
# from itertools import groupby

# def test_table():
#     data_table = pd.DataFrame({'Shelf':(['1. Knowledge']*11 + ['2. Practices']*2 + ['3. Attitudes']*1),
#                                'Questions':['1','2','3','4','5','6','7','8', '9', '10', '11', '12', '13', '14'],
#                                'Quantity':[10,20,5,6,4,7,2,1, 4, 3, 10, 11, 13, 5]
#                                })
#     return data_table

# def add_line(ax, xpos, ypos):
#     line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
#                       transform=ax.transAxes, color='black')
#     line.set_clip_on(False)
#     ax.add_line(line)

# def label_len(my_index,level):
#     labels = my_index.get_level_values(level)
#     return [(k, sum(1 for i in g)) for k,g in groupby(labels)]

# def label_group_bar_table(ax, df):
#     ypos = -.1
#     scale = 1./df.index.size
#     for level in range(df.index.nlevels)[::-1]:
#         pos = 0
#         for label, rpos in label_len(df.index,level):
#             lxpos = (pos + .5 * rpos)*scale
#             ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
#             add_line(ax, pos*scale, ypos)
#             pos += rpos
#         add_line(ax, pos*scale , ypos)
#         ypos -= .1

# df = test_table().groupby(['Shelf','Questions']).sum()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# df.plot(kind='bar',stacked=True,ax=fig.gca())
# #Below 3 lines remove default labels


# labels = ['' for item in ax.get_xticklabels()]
# ax.set_xticklabels(labels)
# ax.set_xlabel('')
# label_group_bar_table(ax, df)
# fig.subplots_adjust(bottom=.1*df.index.nlevels)
# plt.show()


# Multivariate Analysis

# Comparing baseline across regions


df = df_2.copy()
# Working with df now

# df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
# Fixes column names

df['pre_knowledge_score'] = df[pre_questions[0:11]].isin({1}).sum(1)
df['post_knowledge_score'] = df[post_questions[0:11]].isin({1}).sum(1)

# Adding a region column


def f(x):
    if (x in zips_chicago):
        return('Chicago')
    elif (x in zips_only_cook):
        return("Cook")
    else:
        return("IL")


df['Region'] = df['Zip Code'].apply(f)

df = df.rename(columns={'Attendance %': 'Attendance', 'Type (Y/T)': 'Type'})
df = df.rename(columns={'Ever Taken FAN': 'prior_FAN', 'Ever Taken Asthma Class': 'prior_asthma'})
df = df.rename(columns={'3 or 4 day program': 'prog_length'})


# Running regression 1
fit = ols('pre_knowledge_score ~ C(Region) + C(Gender) + C(Race) + C(Grade) + C(prior_FAN) + C(prior_asthma)', data=df).fit()
fit.summary()

# Age and Grade highly correlated - considered only Grade
# Chicago Vs Cook baseline significantly different (p < 0.001)
# Chicago Vs IL less significant (p = 0.021)


# outcome, main independent, covariates - STATA
# covariates vs independent variables


# Running regression 2
fit = ols('pre_knowledge_score ~ C(prior_FAN) + C(Gender) + C(Race) + C(Grade) \
          + C(Region) + C(prior_asthma)', data=df).fit()
fit.summary()

# Same covariates as regression 1?
# prior FAN attendees score 0.94 points higher on baseline knowledge (p < 0.001)


# Running regression 3
df['improvement_knowledge'] = df['post_knowledge_score'] - df['pre_knowledge_score']
fit = ols('improvement_knowledge ~ C(prior_FAN) + C(Gender) + C(Race) + C(Grade) + C(Region) \
          + C(prior_asthma) + C(prog_length) + C(Type) + pre_knowledge_score + Attendance', data=df).fit()
fit.summary()

chicago = df[df.Region == 'Chicago']
cook = df[df.Region == 'Cook']
illinois = df[df.Region == 'IL']

fit = ols('improvement_knowledge ~ C(prior_FAN) + C(Gender) + C(Race) + C(Grade) + C(prior_asthma) \
          + C(prog_length) + C(Type) + pre_knowledge_score + Attendance', data=chicago).fit()
fit.summary()

fit = ols('improvement_knowledge ~ C(prior_FAN) + C(Gender) + C(Race) + C(Grade) + C(prior_asthma) + \
          C(prog_length) + C(Type) + pre_knowledge_score + Attendance', data=cook).fit()
fit.summary()

fit = ols('improvement_knowledge ~ C(prior_FAN) + C(Gender) + C(Race) + C(Grade) + C(prior_asthma) + \
          C(prog_length) + C(Type) + pre_knowledge_score + Attendance', data=illinois).fit()
fit.summary()

# regression 4

# first time vs repeat

non_repeat = df[df.prior_FAN == 2]
repeat = df[df.prior_FAN == 1]

fit = ols('improvement_knowledge ~ C(Region) + C(Gender) + C(Race) + C(Grade) + C(prior_asthma) + \
          C(prog_length) + C(Type) + pre_knowledge_score + Attendance', data=repeat).fit()
fit.summary()

fit = ols('improvement_knowledge ~ C(Region) + C(Gender) + C(Race) + C(Grade) + C(prior_asthma) + \
          C(prog_length) + C(Type) + pre_knowledge_score + Attendance', data=non_repeat).fit()
fit.summary()


# Youth vs Teens

youth = df[df.Type == 'Y']
teen = df[df.Type == 'T']

fit = ols('improvement_knowledge ~ C(prior_FAN) + C(Region) + C(Gender) + C(Race) + C(Grade) + \
          C(prior_asthma) + C(prog_length) + pre_knowledge_score + Attendance', data=youth).fit()
fit.summary()

fit = ols('improvement_knowledge ~ C(prior_FAN) + C(Region) + C(Gender) + C(Race) + C(Grade) + \
          C(prior_asthma) + C(prog_length) + pre_knowledge_score + Attendance', data=teen).fit()
fit.summary()


######
# BOXPLOTS
######


# creating a new df with knowledge scores in long format

df_long = df[['pre_knowledge_score', 'post_knowledge_score', 'Region']]
df_long = df_long.rename(columns={'pre_knowledge_score': 'Pre', 'post_knowledge_score': 'Post'})
# df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
df_long = df_long.melt(id_vars=['Region'])
# Plotting boxplots

# region

sns.set()
sns.set_style(style='ticks')
colors = ["white", "grey", "medium green"]
plt.figure(figsize=(6, 6))

ax = sns.boxplot(x="Region", y='value', width=0.5, hue="variable", palette=sns.xkcd_palette(colors),
                 linewidth=0.5, data=df_long, showfliers=False, order=['Chicago', 'Cook', 'IL'])


# Change line colors to black
plt.setp(ax.artists, edgecolor='k')
plt.setp(ax.lines, color='k')

ax.set_ylabel('Knowledge Score', fontsize=14)
ax.set_xlabel('Region', fontsize=14)
ax.set_title('Fight Asthma Now: Improvement in Knowledge', fontsize=17, pad=20)


plt.legend(bbox_to_anchor=(1, 0.6),
           bbox_transform=plt.gcf().transFigure)


plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], fontsize=12)
ax.set_xticklabels(['Chicago', 'Suburban Cook County', 'Rest of IL'], fontsize=12)

# plt.subplots_adjust(top=0.85)

sns.despine(top=True, right=True)
plt.savefig(os.path.join(path, 'paired-t region'), dpi=300)
plt.close()


# Program Type


df_long = df[['pre_knowledge_score', 'post_knowledge_score', 'Type']]
df_long = df_long.rename(columns={'pre_knowledge_score': 'Pre', 'post_knowledge_score': 'Post'})
df_long = df_long.melt(id_vars=['Type'])


sns.set()
sns.set_style(style='ticks')
colors = ["white", "grey", "medium green"]
plt.figure(figsize=(6, 6))

ax = sns.boxplot(x="Type", y='value', width=0.5, hue="variable", palette=sns.xkcd_palette(colors),
                 linewidth=0.5, data=df_long, order=['Y', 'T'], showfliers=False)


# Change line colors to black
plt.setp(ax.artists, edgecolor='k')
plt.setp(ax.lines, color='k')

ax.set_ylabel('Knowledge Score', fontsize=14)
ax.set_xlabel('Program Type', fontsize=14)
ax.set_title('Fight Asthma Now: Improvement in Knowledge', fontsize=17, pad=20)


plt.legend(bbox_to_anchor=(1, 0.6),
           bbox_transform=plt.gcf().transFigure)


plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], fontsize=12)
ax.set_xticklabels(['Youth', 'Teen'], fontsize=12)

# plt.subplots_adjust(top=0.85)

sns.despine(top=True, right=True)
plt.savefig(os.path.join(path, 'paired-t prog'), dpi=300)
plt.close()


# Repeat vs First time


df_long = df[['pre_knowledge_score', 'post_knowledge_score', 'prior_FAN']]
df_long = df_long.rename(columns={'pre_knowledge_score': 'Pre', 'post_knowledge_score': 'Post'})
df_long = df_long.melt(id_vars=['prior_FAN'])


sns.set()
sns.set_style(style='ticks')
colors = ["white", "grey", "medium green"]
plt.figure(figsize=(6, 6))


ax = sns.boxplot(x="prior_FAN", y='value', width=0.5, hue="variable", palette=sns.xkcd_palette(colors),
                 linewidth=0.5, data=df_long, order=[2., 1.], showfliers=False)

# plt.setp(ax.patches, linewidth = 0.5)

# Change line colors to black
plt.setp(ax.artists, edgecolor='k')
plt.setp(ax.lines, color='k')

ax.set_ylabel('Knowledge Score', fontsize=14)
ax.set_xlabel('Prior FAN Experience', fontsize=14)
ax.set_title('Fight Asthma Now: Improvement in Knowledge', fontsize=17, pad=20)


plt.legend(bbox_to_anchor=(1, 0.6),
           bbox_transform=plt.gcf().transFigure)


plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], fontsize=12)
ax.set_xticklabels(['First-time Attendee', 'Repeat Attendee'], fontsize=12)


# plt.subplots_adjust(top=0.85)

sns.despine(top=True, right=True)
plt.savefig(os.path.join(path, 'paired-t fan box'), dpi=300)
plt.close()


# Unpaired-t boxplots

# Program Type

df_long = df[['pre_knowledge_score', 'Type']]
df_long = df_long.rename(columns={'pre_knowledge_score': 'Pre', 'post_knowledge_score': 'Post'})


sns.set()
sns.set_style(style='ticks')
colors = ["white", "white", "medium green"]
plt.figure(figsize=(6, 6))

ax = sns.boxplot(x="Type", y='Pre', width=0.5, palette=sns.xkcd_palette(colors),
                 linewidth=0.5, data=df_long, order=['Y', 'T'], showfliers=False)


# Change line colors to black
plt.setp(ax.artists, edgecolor='k')
plt.setp(ax.lines, color='k')

# plt.setp(ax.patches, linewidth = 0.5)
ax.set_ylabel('Knowledge Score', fontsize=14)
ax.set_xlabel('Program Type', fontsize=14)
ax.set_title('Fight Asthma Now: Comparison of Baseline Knowledge', fontsize=17, pad=20)

plt.legend(bbox_to_anchor=(1, 0.6),
           bbox_transform=plt.gcf().transFigure)

plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], fontsize=12)
ax.set_xticklabels(['Youth', 'Teen'], fontsize=12)

# plt.subplots_adjust(top=0.85)

sns.despine(top=True, right=True)
plt.savefig(os.path.join(path, 'unpaired-t prog'), dpi=300)
plt.close()


# Region
df_long = df[['pre_knowledge_score', 'post_knowledge_score', 'Region']]
df_long = df_long.rename(columns={'pre_knowledge_score': 'Pre', 'post_knowledge_score': 'Post'})
# Plotting boxplots


sns.set()
sns.set_style(style='ticks')
colors = ["white", "white", "white"]
plt.figure(figsize=(6, 6))
# plt.figure(figsize=(10,20))

ax = sns.boxplot(x="Region", y='Pre', width=0.5, palette=sns.xkcd_palette(colors),
                 linewidth=0.5, data=df_long, showfliers=False, order=['Chicago', 'Cook', 'IL'])


# Change line colors to black
plt.setp(ax.artists, edgecolor='k')
plt.setp(ax.lines, color='k')

ax.set_ylabel('Knowledge Score', fontsize=14)
ax.set_xlabel('Region', fontsize=14)
ax.set_title('Fight Asthma Now: Comparison of Baseline Knowledge', fontsize=17, pad=20)

plt.legend(bbox_to_anchor=(1, 0.6),
           bbox_transform=plt.gcf().transFigure)

plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], fontsize=12)
ax.set_xticklabels(['Chicago', 'Suburban Cook County', 'Rest of IL'], fontsize=12)

# plt.subplots_adjust(top=0.85)

sns.despine(top=True, right=True)
plt.savefig(os.path.join(path, 'unpaired-t region'), dpi=300)
plt.close()


# Repeat vs First time


df_long = df[['pre_knowledge_score', 'post_knowledge_score', 'prior_FAN']]
df_long = df_long.rename(columns={'pre_knowledge_score': 'Pre', 'post_knowledge_score': 'Post'})


sns.set()
sns.set_style(style='ticks')
colors = ["white", "white", "medium green"]
plt.figure(figsize=(6, 6))


ax = sns.boxplot(x="prior_FAN", y='Pre', width=0.5, palette=sns.xkcd_palette(colors),
                 linewidth=0.5, data=df_long, order=[2., 1.], showfliers=False)

# plt.setp(ax.patches, linewidth = 0.5)

# Change line colors to black
plt.setp(ax.artists, edgecolor='k')
plt.setp(ax.lines, color='k')

ax.set_ylabel('Knowledge Score', fontsize=14)
ax.set_xlabel('Prior FAN Experience', fontsize=14)
ax.set_title('Fight Asthma Now: Comparison of Baseline Knowledge', fontsize=17, pad=20)

plt.legend(bbox_to_anchor=(1, 0.6),
           bbox_transform=plt.gcf().transFigure)

plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], fontsize=12)
ax.set_xticklabels(['First-time Attendee', 'Repeat Attendee'], fontsize=12)


# plt.subplots_adjust(top=0.85)

sns.despine(top=True, right=True)
plt.savefig(os.path.join(path, 'unpaired-t fan'), dpi=300)
plt.close()


# Overall boxplot

df_long = df[['ID #', 'pre_knowledge_score', 'post_knowledge_score']]
df_long = df_long.rename(columns={'pre_knowledge_score': 'Pre', 'post_knowledge_score': 'Post'})
df_long = df_long.melt(id_vars=['ID #'])

sns.set()
sns.set_style(style='ticks')
colors = ["white", "grey", "medium green"]
plt.figure(figsize=(6, 6))

ax = sns.boxplot(x="variable", y='value', width=0.5, palette=sns.xkcd_palette(colors),
                 linewidth=0.5, data=df_long, showfliers=False)


# Change line colors to black
plt.setp(ax.artists, edgecolor='k')
plt.setp(ax.lines, color='k')

# plt.setp(ax.patches, linewidth = 0.5)
ax.set_ylabel('Knowledge Score', fontsize=14)
ax.set_xlabel('Score Type', fontsize=14)
ax.set_title('Fight Asthma Now: Change in Knowledge', fontsize=18, pad=20)

plt.legend(bbox_to_anchor=(1, 0.6),
           bbox_transform=plt.gcf().transFigure)


plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], fontsize=12)
plt.xticks(fontsize=12)

# plt.subplots_adjust(top=0.85)

sns.despine(top=True, right=True)
plt.savefig(os.path.join(path, 'overall fan'), dpi=300)
plt.close()


# Statistical Testing

# Overall


# Paired t tests, testing for normality using the Shapiro-Wilk test


stats.shapiro(df['pre_knowledge_score'])
# p-val = 2.498508362185737e-25, normality rejected

stats.shapiro(df['post_knowledge_score'])
# p-val = 0, normality rejected

# Can't use the paired-t test, use the Wilcoxon Sign-Ranked Test instead
stats.wilcoxon(df['pre_knowledge_score'], df['post_knowledge_score'])
# p-val = 6.262864384682711e-267, highly significant

# df.post_knowledge_score.mean() - df.pre_knowledge_score.mean()
# df[df.Region == 'Chicago'].pre_knowledge_score.mean()
# df[df.Region == 'IL'].pre_knowledge_score.mean()
# df[df.Region == 'Cook'].pre_knowledge_score.mean()
# df[df.prior_FAN == 1].pre_knowledge_score.mean() - df[df.prior_FAN == 2].pre_knowledge_score.mean()


#######
# Mixed effects regression
#######


multi_cols = ["ID #", "prog_length", "Type", "prior_asthma", "prior_FAN", "Gender", "Race", "Age",
              "Grade", "Year", "Attendance", "Region", "pre_knowledge_score", "post_knowledge_score"]

df_multi = df[multi_cols].copy()

df_multi = df_multi.rename(
    columns={'pre_knowledge_score': 'score_0', 'post_knowledge_score': 'score_1'})
df_multi['Index'] = range(1, len(df_multi) + 1)
df_multi = df_multi.set_index('Index')

df_multi = pd.wide_to_long(df_multi.reset_index(), stubnames='score_',
                           i='Index', j='Time')

df_multi = df_multi.rename(columns={'score_': 'knowledge_score'})

# Manual Verification
df_multi[df_multi['ID #'] == 'UCMO313']
# Success!


data = df_multi.reset_index()
data = data.dropna()  # Remove NAs
zzzz = data
# Region


data.to_csv(os.path.join(path, 'data.csv'))


md = smf.mixedlm("knowledge_score ~ C(prior_FAN)+C(prior_asthma)+Type+Age+C(Race)+C(Gender)+Time+Attendance+prog_length",
                 data, groups=data["Region"], re_formula="~Time")
mdf = md.fit()
print(mdf.summary())


# Instead of running a t-test on the pre and post knowledge scores,
# run a test for each individual question (score out of 1)
# You get a score of 1 if you answer correctly, 0 otherwise.

# Paired t tests, testing for normality using the Shapiro-Wilk test


kpa_questions = ["1: Warning Signs.1", "2: Smoking Trigger.1",
                 "3: Spacer.1", "4: Slow Breath.1", "5: Squeezing.1", "6: 10-15.1", "7: Swelling/Snot.1",
                 "8: Everyday.1", "9: Calm.1", "10: Exercise.1", "11: Life-long.1", "12: Talk to Adults.1",
                 "13: Inhaler at School.1", "14: Comfortable Carry.1", "1: Warning Signs.2",
                 "2: Smoking Trigger.2", "3: Spacer.2", "4: Slow Breath.2", "5: Squeezing.2", "6: 10-15.2", "7: Swelling/Snot.2", "8: Everyday.2", "9: Calm.2", "10: Exercise.2",
                 "11: Life-long.2", "13: Talk AAP.2", "14: Inhaler at School.2", "15: Comfortable Carry.2"]


df_questions = df_2[kpa_questions].copy()
# Changing 2s, 3s to 0s
# Dropping 999,NAs

df_questions = df_questions.replace(to_replace=2, value=0)
df_questions = df_questions.replace(to_replace=3, value=0)

df_questions = df_questions.dropna()
len(df_questions)  # 2566 entries left


# Q1
stats.shapiro(df_questions['1: Warning Signs.1'])
stats.shapiro(df_questions['1: Warning Signs.2'])

# Can't use the paired-t test, use the Wilcoxon Sign-Ranked Test instead
stats.wilcoxon(df_questions['1: Warning Signs.1'], df_questions['1: Warning Signs.2'])
# p-val = 9.429265594120709e-153, highly significant


# Q9
stats.shapiro(df_questions['9: Calm.1'])
stats.shapiro(df_questions['9: Calm.2'])

# Can't use the paired-t test, use the Wilcoxon Sign-Ranked Test instead
stats.wilcoxon(df_questions['9: Calm.1'], df_questions['9: Calm.2'])
# p-val = 1.9764195471703374e-08, highly significant


# Q14
stats.shapiro(df_questions['14: Comfortable Carry.1'])
stats.shapiro(df_questions['15: Comfortable Carry.2'])

# Can't use the paired-t test, use the Wilcoxon Sign-Ranked Test instead
stats.wilcoxon(df_questions['14: Comfortable Carry.1'], df_questions['15: Comfortable Carry.2'])
# p-val = 2.2337570634266743e-10, highly significant


stats.shapiro(df['pre_knowledge_score'])
# seed the random number generator
seed(1)

# normality test
result = anderson(df['post_knowledge_score'])
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


# Working with df to create Hispanic and Black variables


# def f(x):
#     if (x in zips_chicago):
#         return('Chicago')
#     elif (x in zips_only_cook):
#         return("Cook")
#     else:
#         return("IL")


# df['Region'] =df['Zip Code'].apply(f)

def black(x):
    if (x == 2):
        return(1)
    else:
        return(0)


def hispanic(x):
    if (x == 3):
        return(1)
    else:
        return(0)


df['Black'] = df['Race'].apply(black)

df['Hispanic'] = df['Race'].apply(hispanic)

#######
# Mixed effects regression with race dummies
#######


multi_cols = ["ID #", "prog_length", "Type", "prior_asthma", "prior_FAN", "Gender", "Age",
              "Grade", "Year", "Attendance", "Region", "pre_knowledge_score", "post_knowledge_score",
              "Black", "Hispanic"]

df_multi = df[multi_cols].copy()

df_multi = df_multi.rename(
    columns={'pre_knowledge_score': 'score_0', 'post_knowledge_score': 'score_1'})
df_multi['Index'] = range(1, len(df_multi) + 1)
df_multi = df_multi.set_index('Index')

df_multi = pd.wide_to_long(df_multi.reset_index(), stubnames='score_',
                           i='Index', j='Time')

df_multi = df_multi.rename(columns={'score_': 'knowledge_score'})

# Manual Verification
df_multi[df_multi['ID #'] == 'UCMO313']
# Success!


data = df_multi.reset_index()
data = data.dropna()  # Remove NAs


md = smf.mixedlm("knowledge_score ~ C(prior_FAN)+C(prior_asthma)+Type+Age+\
                C(Black)+C(Hispanic)+C(Gender)+Time+Attendance+prog_length", data,
                 groups=data["Region"], re_formula="~Time")
mdf = md.fit()
print(mdf.summary())


# Running regression 1
fit = ols('pre_knowledge_score ~ C(Region) + C(Gender) + C(Race) + C(Grade) + C(prior_FAN) + C(prior_asthma)', data=df).fit()
fit.summary()

# Age and Grade highly correlated - considered only Grade
# Chicago Vs Cook baseline significantly different (p < 0.001)
# Chicago Vs IL less significant (p = 0.021)


# outcome, main independent, covariates - STATA
# covariates vs independent variables


# Running regression 2
fit = ols('pre_knowledge_score ~ C(prior_FAN) + C(Gender) + C(Race) + C(Grade) \
          + C(Region) + C(prior_asthma)', data=df).fit()
fit.summary()


df[df.Black == 1]['post_knowledge_score'].mean()
df[df.Hispanic == 1]['post_knowledge_score'].mean()
df[(df.Black == 0)]['pre_knowledge_score'].mean()
df[(df.Hispanic == 0)]['pre_knowledge_score'].mean()


df[df.Black == 1]['post_knowledge_score'].mean()
df[df.Hispanic == 1]['post_knowledge_score'].mean()
df[(df.Black == 0) & (df.Hispanic == 0)]['pre_knowledge_score'].mean()


# Explore Race differences


# Miscellaneous

repeat[repeat.Year == '2019-20']['pre_knowledge_score'].mean()

df[df.Year == '2019-20']['post_knowledge_score'].mean() - df[df.Year ==
                                                             '2019-20']['pre_knowledge_score'].mean()


stats.wilcoxon(illinois[illinois.Year == '2019-20']['pre_knowledge_score'],
               illinois[illinois.Year == '2019-20']['post_knowledge_score'])  # paired

illinois[illinois.Year == '2019-20']['post_knowledge_score'].mean() - illinois[illinois.Year ==
                                                                               '2019-20']['pre_knowledge_score'].mean()

stats.levene(cook[cook.Year == '2019-20']['pre_knowledge_score'],
             illinois[illinois.Year == '2019-20']['pre_knowledge_score'])  # unpaired

non_repeat['post_knowledge_score'].mean() - non_repeat['pre_knowledge_score'].mean()
