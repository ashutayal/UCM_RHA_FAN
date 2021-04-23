#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:09:23 2021

@author: ashutayal
"""

from statsmodels.stats.contingency_tables import mcnemar as mcnemars_stat
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set working directory to location of csvs
path = '/Users/ashutayal/RHA_letter'

# df_2017 contains 2017-18, df_2018 contains 2018-19 data and so on
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

df.to_csv(os.path.join(path, 'df-raw.csv'))


# CLEAN UP

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


# Type missing to 999
df.loc[df['Type (Y/T)'] == ".", ['Type (Y/T)']] = 999


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


# Export as csv
df.to_csv(os.path.join(path, 'df-clean.csv'))


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

    return(test)


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
    v13 = pd.pivot_table(df_3, index=['13: Talk AAP.2'], values='ID #', aggfunc=len)
    v14 = pd.pivot_table(df_3, index=['14: Inhaler at School.2'], values='ID #', aggfunc=len)
    v15 = pd.pivot_table(df_3, index=['15: Comfortable Carry.2'], values='ID #', aggfunc=len)
    trial = pd.concat([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v13, v14, v15], axis=1)
    trial = trial.transpose().reset_index(drop=True)
    return(trial)


def mcnemar(df):
    final_df = df.copy()
    final_df[cols] = final_df[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
    final_df[cols] = final_df[cols].replace({3: 2, '3': 2})
    result = pd.DataFrame(columns=["Questions", "Outcome 1", "Outcome 2", "Outcome 3", "Outcome 4"])

    for pre, post in zip(pre_questions, post_questions):
        final_df = final_df[(final_df[pre].isin({1, 2})) & (final_df[post].isin({1, 2}))]
        o1 = len(final_df[(final_df[pre].isin({1})) & (final_df[post].isin({1}))])
        o3 = len(final_df[(final_df[pre].isin({2})) & (final_df[post].isin({1}))])
        o2 = len(final_df[(final_df[pre].isin({1})) & (final_df[post].isin({2}))])
        o4 = len(final_df[(final_df[pre].isin({2})) & (final_df[post].isin({2}))])
        result = result.append(
            {"Questions": pre[:-2], "Outcome 1": o1, "Outcome 2": o2, "Outcome 3": o3, "Outcome 4": o4}, ignore_index=True)

        # pre_correct = len(final_df[final_df[pre] == 1])

    return(result)


# McNemar's test


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


# df_2 used for means (nas are excluded from means)
df_2 = df.copy()
df_2[cols] = df_2[cols].replace({999: np.nan, 0: np.nan, 9999: np.nan, '999': np.nan})

# df_3 used for counts
# Here 0s, 999s, 9999s are treated alike
df_3 = df.copy()
df_3[cols] = df_3[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
df_3['Race'] = df_3['Race'].replace({7: 999})  # 7 is no answer


def final_pivot(df):
    final_df = df.copy()
    final_df[cols] = final_df[cols].replace({0: 999, 9999: 999, '999': 999, '0': 999})
    final_df[cols] = final_df[cols].replace({3: 2, '3': 2})
    test = pd.DataFrame(columns=["Questions", "N", "Pre Correct",
                                 "Pre Incorrect", "Post Correct", "Post Incorrect"])

    for pre, post in zip(pre_questions, post_questions):

        final_df = final_df[(final_df[pre].isin({1, 2})) & (final_df[post].isin({1, 2}))]
        pre_correct = len(final_df[final_df[pre] == 1])
        post_correct = len(final_df[final_df[post] == 1])
        pre_incorrect = len(final_df[final_df[pre] == 2])
        post_incorrect = len(final_df[final_df[post] == 2])
        if (pre_correct+pre_incorrect == post_correct+post_incorrect):  # checks if totals consistent
            N = pre_correct+pre_incorrect
        else:
            N = 0
        test = test.append({"Questions": pre[:-2], "N": N, "Pre Correct": pre_correct, "Pre Incorrect": pre_incorrect,
                            "Post Correct": post_correct, "Post Incorrect": post_incorrect}, ignore_index=True)

    return(test)


df_all_ques_only = df_3.copy()
for ques in pre_questions+post_questions:
    df_all_ques_only = df_all_ques_only[df_all_ques_only[ques] != 999]

df_all_ques_only['pre_knowledge_score'] = df_all_ques_only[pre_questions[0:11]].isin({1}).sum(1)
df_all_ques_only['post_knowledge_score'] = df_all_ques_only[post_questions[0:11]].isin({1}).sum(1)


# separate based on prior asthma education

repeat_all = df_all_ques_only[(df_all_ques_only['Ever Taken FAN'] == 1) | (
    df_all_ques_only['Ever Taken Asthma Class'] == 1)].reset_index(drop=True)
non_repeat_all = df_all_ques_only[(df_all_ques_only['Ever Taken FAN'] != 1) & (
    df_all_ques_only['Ever Taken Asthma Class'] != 1)].reset_index(drop=True)

repeat_any = df_3[(df_3['Ever Taken FAN'] == 1) | (
    df_3['Ever Taken Asthma Class'] == 1)].reset_index(drop=True)
non_repeat_any = df_3[(df_3['Ever Taken FAN'] != 1) & (
    df_3['Ever Taken Asthma Class'] != 1)].reset_index(drop=True)

final_table = final_pivot(df_3)
repeat_overall = final_pivot(repeat_any)
non_repeat_overall = final_pivot(non_repeat_any)

# non_repeat_any: this includes first time participants who answered at least one on pre and on post
# non_repeat_all: this includes first time participants who answered all on pre and on post
# Reason: scores only calculated for the participants who answered all questions on pre and post, so important to separate the two

# Statistical tests


# Paired tests - Wilcoxon

# stats.wilcoxon(non_repeat['pre_knowledge_score'], non_repeat['post_knowledge_score'])
# df_all_ques_only['post_knowledge_score'].mean() - df_all_ques_only['pre_knowledge_score'].mean()
# non_repeat['pre_knowledge_score'].describe()

# Unpaired/2 sample tests - Mann whitney U
# stats.mannwhitneyu(repeat['pre_knowledge_score'], non_repeat['pre_knowledge_score'])
# repeat['pre_knowledge_score'].mean() - non_repeat['pre_knowledge_score'].mean()
# repeat['post_knowledge_score'].mean() - non_repeat['post_knowledge_score'].mean()


# McNemar's
overall_mcn_pvals = mc_p_vals(df_3)
repeat_mcn_pvals = mc_p_vals(repeat_any)
non_repeat_mcn_pvals = mc_p_vals(non_repeat_any)


# separate based on prior FAN

repeat_all = df_all_ques_only[df_all_ques_only['Ever Taken FAN'] == 1].reset_index(drop=True)
non_repeat_all = df_all_ques_only[df_all_ques_only['Ever Taken FAN'] != 1].reset_index(drop=True)

repeat_any = df_3[df_3['Ever Taken FAN'] == 1].reset_index(drop=True)
non_repeat_any = df_3[df_3['Ever Taken FAN'] != 1].reset_index(drop=True)

final_table = final_pivot(df_3)
repeat_overall = final_pivot(repeat_any)
non_repeat_overall = final_pivot(non_repeat_any)


# Statistical tests


# Paired tests - Wilcoxon

# stats.wilcoxon(non_repeat['pre_knowledge_score'], non_repeat['post_knowledge_score'])
# df_all_ques_only['post_knowledge_score'].mean() - df_all_ques_only['pre_knowledge_score'].mean()
# non_repeat['pre_knowledge_score'].describe()

# Unpaired/2 sample tests - Mann whitney U

# stats.mannwhitneyu(repeat['pre_knowledge_score'], non_repeat['pre_knowledge_score'])
# repeat['pre_knowledge_score'].mean() - non_repeat['pre_knowledge_score'].mean()
# repeat['post_knowledge_score'].mean() - non_repeat['post_knowledge_score'].mean()


# McNemar's
overall_mcn_pvals = mc_p_vals(df_3)
repeat_mcn_pvals = mc_p_vals(repeat_any)
non_repeat_mcn_pvals = mc_p_vals(non_repeat_any)
