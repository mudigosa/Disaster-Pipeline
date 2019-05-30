{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import pandas as pd\n",
    "from sqlalchemy import create_engine\n",
    "\n",
    "def load_data(messages_filepath, categories_filepath):\n",
    "    '''\n",
    "    input:\n",
    "        messages_filepath: The path of messages dataset.\n",
    "        categories_filepath: The path of categories dataset.\n",
    "    output:\n",
    "        df: The merged dataset\n",
    "    '''\n",
    "    disastermessages = pd.read_csv('disaster_messages.csv')\n",
    "    disastermessages.head()\n",
    "    # load categories dataset\n",
    "    disastercategories = pd.read_csv('disaster_categories.csv')\n",
    "    disastercategories.head()\n",
    "    df = pd.merge(disastermessages, disastercategories, left_on='id', right_on='id', how='outer')\n",
    "    return df\n",
    "\n",
    "def clean_data(df):\n",
    "    '''\n",
    "    input:\n",
    "        df: The merged dataset in previous step.\n",
    "    output:\n",
    "        df: Dataset after cleaning.\n",
    "    '''\n",
    "    disastercategories = df.disastercategories.str.split(';', expand = True)\n",
    "    # select the first row of the categories dataframe\n",
    "    row = disastercategories.iloc[0,:]\n",
    "\n",
    "    # use this row to extract a list of new column names for categories.\n",
    "    # one way is to apply a lambda function that takes everything \n",
    "    # up to the second to last character of each string with slicing\n",
    "    disastercategory_colnames = row.apply(lambda x:x[:-2])\n",
    "    print(disastercategory_colnames)\n",
    "    disastercategories.columns = category_colnames\n",
    "    for column in disastercategories:\n",
    "    # set each value to be the last character of the string\n",
    "        disastercategories[column] = disastercategories[column].str[-1]\n",
    "    \n",
    "    # convert column from string to numeric\n",
    "    disastercategories[column] = disastercategories[column].astype(np.int)\n",
    "    disastercategories.head()\n",
    "    df.drop('categories', axis = 1, inplace = True)\n",
    "    df = pd.concat([df, categories], axis = 1)\n",
    "    # drop the original categories column from `df`\n",
    "    df = df.drop('categories',axis=1)\n",
    "\n",
    "    df.head()\n",
    "    # check number of duplicates\n",
    "    print('Number of duplicated rows: {} out of {} samples'.format(df.duplicated().sum(),df.shape[0]))\n",
    "    df.drop_duplicates(subset = 'id', inplace = True)\n",
    "    return df\n",
    "\n",
    "def save_data(df, database_filepath):\n",
    "    engine = create_engine('sqlite:///DisasterResponse.db')\n",
    "    df.to_sql('df', engine, index=False)\n",
    "\n",
    "def main():\n",
    "    if len(sys.argv) == 4:\n",
    "\n",
    "        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]\n",
    "\n",
    "        print('Loading data...\\n    MESSAGES: {}\\n    CATEGORIES: {}'\n",
    "              .format(messages_filepath, categories_filepath))\n",
    "        df = load_data(messages_filepath, categories_filepath)\n",
    "\n",
    "        print('Cleaning data...')\n",
    "        df = clean_data(df)\n",
    "\n",
    "        print('Saving data...\\n    DATABASE: {}'.format(database_filepath))\n",
    "        save_data(df)\n",
    "\n",
    "        print('Cleaned data saved to database!')\n",
    "\n",
    "    else:\n",
    "        print('Please provide the filepaths of the messages and categories '\\\n",
    "              'datasets as the first and second argument respectively, as '\\\n",
    "              'well as the filepath of the database to save the cleaned data '\\\n",
    "              'to as the third argument. \\n\\nExample: python process_data.py '\\\n",
    "              'disaster_messages.csv disaster_categories.csv '\\\n",
    "              'DisasterResponse.db')\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}