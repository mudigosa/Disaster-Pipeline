import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    input:
        messages_filepath: The path of messages dataset.
        categories_filepath: The path of categories dataset.
    output:
        df: The merged dataset
    '''
    disastermessages = pd.read_csv('disaster_messages.csv')
    disastermessages.head()
    # load categories dataset
    disastercategories = pd.read_csv('disaster_categories.csv')
    disastercategories.head()
    df = pd.merge(disastermessages, disastercategories, left_on='id', right_on='id', how='outer')
    return df

def clean_data(df):
    '''
    input:
        df: The merged dataset in previous step.
    output:
        df: Dataset after cleaning.
    '''
    disastercategories = df.categories.str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = disastercategories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    disastercategory_colnames = row.apply(lambda x:x[:-2])
    print(disastercategory_colnames)
    disastercategories.columns = category_colnames
    for column in disastercategories:
    # set each value to be the last character of the string
        disastercategories[column] = disastercategories[column].str[-1]
    
    # convert column from string to numeric
    disastercategories[column] = disastercategories[column].astype(np.int)
    disastercategories.head()
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)

    df.head()
    # check number of duplicates
    print('Number of duplicated rows: {} out of {} samples'.format(df.duplicated().sum(),df.shape[0]))
    df.drop_duplicates(subset = 'id', inplace = True)
    return df

def save_data(df):
    engine = create_engine('sqlite:///disastermessages.db')
    df.to_sql('df', engine, index=False)

def main():
        df = load_data()
        print('Cleaning data...')
        df = clean_data(df)
        save_data(df)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
main()
