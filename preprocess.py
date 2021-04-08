import pandas as pd




def remove_state_with_hyphens(address):
    states = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'hi', 'ia', 'id', 'il', 'in',
    'ks', 'ky', 'la', 'ma', 'md', 'me', 'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nj',
    'nm', 'nv', 'ny', 'oh','ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa',
    'wi', 'wv', 'wy']
    for state in states:
        if "us-"+state:
            address = address.replace("us-"+state, state)
        if "US-"+state.upper():
            address = address.replace("US-"+state.upper(), state.upper())
    return address

def replace_comma(dataframe, field_name):
    dataframe[field_name] = dataframe[field_name].str.replace(",", " , ")


def remove_at(dataframe, field_name):
    dataframe[field_name] = dataframe[field_name].str.replace("@", " ")
    return dataframe


def remove_dollar(dataframe, field_name):
    dataframe[field_name] = dataframe[field_name].str.replace("$", " ")
    return dataframe


def remove_hash(dataframe, field_name):
    dataframe[field_name] = dataframe[field_name].str.replace("#", " ")
    return dataframe


def remove_percent(dataframe, field_name):
    dataframe[field_name] = dataframe[field_name].str.replace("%", " ")
    return dataframe


def remove_care_of(dataframe, field_name):
    dataframe[field_name] = dataframe[field_name].str.replace("c/o", " ")
    return dataframe


def remove_attn(dataframe, field_name):
    dataframe[field_name] = dataframe[field_name].str.replace("attn :", " ")
    dataframe[field_name] = dataframe[field_name].str.replace("Attn :", " ")
    return dataframe


def remove_multiple_white_spaces(dataframe, field_name):
    dataframe[field_name] = dataframe[field_name].str.replace(" +", " ", regex=True)
    return dataframe


def strip_whitespaces(dataframe):
    dataframe[dataframe.columns] = dataframe.apply(lambda s: s.str.strip() if type(s) == str else s)
    return dataframe




def clean_data(text):
    mydict = {'Raw Data':[text]}
    df_std = pd.DataFrame(mydict)
    df_std['Raw Data'] = df_std.apply(lambda row: remove_state_with_hyphens(row['Raw Data']), axis=1)
    df_std['Raw Data'] = df_std['Raw Data'].str.replace("united states of america", " ")
    df_std['Raw Data'] = df_std['Raw Data'].str.replace("United States of America", " ")
    df_std['Raw Data'] = df_std['Raw Data'].str.replace("united states of ame", " ")
    df_std['Raw Data'] = df_std['Raw Data'].str.replace("United States", " ")
    df_std['Raw Data'] = df_std['Raw Data'].str.replace("United States of Ame", " ")
    df_std['Raw Data'] = df_std['Raw Data'].str.replace("UNITED STATES OF AMERICA", " ")
    df_std['Raw Data'] = df_std['Raw Data'].str.replace("USA", " ")

    replace_comma(df_std, "Raw Data")
    remove_hash(df_std, "Raw Data")
    remove_dollar(df_std, "Raw Data")
    remove_percent(df_std, "Raw Data")
    remove_at(df_std, "Raw Data")
    remove_multiple_white_spaces(df_std, "Raw Data")
    remove_care_of(df_std, "Raw Data")
    remove_attn(df_std, "Raw Data")

    return df_std['Raw Data'][0]