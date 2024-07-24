import pandas as pd
import json
import re
from fuzzywuzzy import process
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_company_name(name):
    """
    Normalizes a company name by performing the following operations:
    - Converts the name to lowercase.
    - Removes suffixes such as 'Ltd', 'Limited', 'Inc', 'Incorporated'.
    - Removes leading/trailing whitespace and periods.
    - Replaces multiple spaces with a single space.
    """
    name = name.lower()  # Convert to lowercase
    name = re.sub(r'\b(ltd|limited|inc|incorporated)\b\.?', '', name)  # Remove suffixes
    name = name.strip().strip('.')  # Trim whitespace and periods
    name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with single space
    return name

def load_data():
    """
    Loads and preprocesses the data from CSV and JSON files.
    - Reads 'NIFTY500.csv' into a DataFrame and normalizes the company names.
    - Reads 'fixed_symbols.json' into a DataFrame and normalizes the company names.
    - Writes unique company names from 'NIFTY500.csv' to 'company_names.txt'.
    - Reads 'stocks_df.csv' into a DataFrame.
    """
    # Load timeline data
    timeline_df = pd.read_csv('./data/NIFTY500.csv')
    timeline_df['Scrip Name'] = timeline_df['Scrip Name'].apply(normalize_company_name)  # Normalize company names
    
    # Load symbols data
    symbols_df = pd.read_json('./fixed_symbols.json', orient='index').reset_index()
    symbols_df.columns = ['Company', 'Symbol']  # Rename columns
    symbols_df['Company'] = symbols_df['Company'].apply(normalize_company_name)  # Normalize company names
    
    # Write unique company names to a file
    with open('company_names.txt', 'w') as f:
        f.write(str(list(timeline_df['Scrip Name'].unique())))
    
    # Load stocks data
    stocks_df = pd.read_csv('./data/stocks_df.csv')

    return timeline_df, symbols_df, stocks_df

def build_index_dict(timeline_df):
    """
    Constructs a dictionary to track index composition over time.
    - Initializes an empty dictionary with the earliest date as the key.
    - Iterates over the DataFrame and updates the dictionary based on 'In' and 'Ex' events.
    """
    index = {}
    timeline = pd.to_datetime(timeline_df["Event Date"], format='%d-%m-%Y')  # Convert dates to datetime
    descs = timeline_df["Description"].to_numpy()  # Convert descriptions to numpy array
    stocks = timeline_df["Scrip Name"].to_numpy()  # Convert company names to numpy array
    
    index[timeline.min()] = []  # Initialize dictionary with earliest date
    
    for i in range(len(timeline_df)):
        current_date = timeline[i]
        if current_date not in index:
            # If current date not in index, copy the previous date's list
            index[current_date] = list(index[max(k for k in index.keys() if k < current_date)])
        
        # Handle 'In' events
        if descs[i].startswith("In"):
            if stocks[i] not in index[current_date]:
                index[current_date].append(stocks[i])
            else:
                logging.warning(f"{current_date} -> {stocks[i]} already exists")
        
        # Handle 'Ex' events
        if descs[i].startswith("Ex"):
            if stocks[i] in index[current_date]:
                index[current_date].remove(stocks[i])
            else:
                logging.warning(f"{current_date} -> {stocks[i]} not found in list")
    return index

def match_company_names(companies, symbols_df):
    """
    Matches company names with their symbols using fuzzy matching.
    - For each company, finds the closest match in the symbols DataFrame.
    - Returns a dictionary mapping company names to their symbols.
    """
    def find_closest_match(name):
        company_names = symbols_df['Company'].tolist()  # List of company names from symbols DataFrame
        match, score = process.extractOne(name, company_names)  # Find closest match using fuzzy matching
        if score >= 90:  # If score is 90 or above, consider it a match
            return symbols_df[symbols_df['Company'] == match]['Symbol'].iloc[0]
        elif score < 90 and score > 80:  # If score is between 80 and 90, log a warning and consider it a match
            logging.warning(f"symbol found for company: {name} with symbol {str(match).upper()} at score {score}")
            return symbols_df[symbols_df['Company'] == match]['Symbol'].iloc[0]
        logging.error(f"No symbol found for company: {name}")  # If score is below 80, log an error
        return None

    return {company: find_closest_match(company) for company in companies}

def get_index_composition_2000_2020(index_dict, symbols_dict):
    """
    Retrieves the index composition from 2010 to 2020.
    - Iterates over the dates and updates the composition based on changes.
    - Creates a list of companies in the index for each date within the specified range.
    """
    start_date = pd.Timestamp('2010-01-01')
    end_date = pd.Timestamp('2020-12-31')
    
    # Sort the index dictionary dates
    index_dates = sorted(index_dict.keys())
    
    # Initialize composition with the earliest date
    current_composition = set(index_dict[index_dates[0]])
    
    composition = {}
    
    # Process all changes up to start_date
    for date in index_dates:
        if date > start_date:
            break
        current_composition = set(index_dict[date])
    
    # Create a date range for all dates between start and end
    all_dates = pd.date_range(start=start_date, end=end_date)
    
    date_idx = next(i for i, d in enumerate(index_dates) if d >= start_date)
    
    for date in all_dates:
        # Update composition if we've reached a new change date
        while date_idx < len(index_dates) and date >= index_dates[date_idx]:
            current_composition = set(index_dict[index_dates[date_idx]])
            date_idx += 1
        
        # Add the current composition to our result
        composition[date] = [
            {
                'Index Name': 'Nifty 500',
                'Event Date': date.strftime('%Y-%m-%d'),
                'Scrip Name': scrip,
                'Description': 'In Index',
                'ticker': symbols_dict.get(scrip, '')
            }
            for scrip in current_composition
        ]
    
    return composition

def main():
    """
    Main function to execute the workflow:
    - Loads data.
    - Builds the index dictionary.
    - Matches company names with symbols.
    - Retrieves index composition for 2010-2020.
    - Flattens the composition dictionary to create an updated timeline DataFrame.
    - Saves the updated timeline to a CSV file.
    """
    # Load data
    timeline_df, symbols_df, stocks_df = load_data()
    
    # Build index dictionary
    index_dict = build_index_dict(timeline_df)
    
    # Get all unique companies in the index
    all_companies = set(timeline_df['Scrip Name'])
    
    # Match company names and find symbols
    symbol_dict = match_company_names(all_companies, symbols_df)
    
    # Get index composition for 2010-2020
    composition_2010_2020 = get_index_composition_2000_2020(index_dict, symbol_dict)
    
    # Flatten the composition dictionary to create the updated timeline
    updated_timeline = []
    for date, companies in composition_2010_2020.items():
        for company in companies:
            updated_timeline.append(company)
    
    updated_df = pd.DataFrame(updated_timeline)
    
    # Print column names for debugging
    print("Columns in updated_df:", updated_df.columns)
    
    # Check if 'Event Date' column exists, if not, use the first column as the date column
    date_column = 'Event Date' if 'Event Date' in updated_df.columns else updated_df.columns[0]
    
    # Sort the dataframe by date
    updated_df = updated_df.sort_values(date_column)
    
    # Save updated timeline to CSV
    updated_df.to_csv('./data/NIFTY500_2010_2020.csv', index=False)
    
    # Log some information about the updated timeline
    logging.info(f"Timeline entries: {len(updated_df)}")
    logging.info(f"Unique dates in timeline: {updated_df[date_column].nunique()}")
    logging.info(f"Symbols matched: {updated_df['ticker'].notna().sum() if 'ticker' in updated_df.columns else 'N/A'}")
    
    # Print first few rows for debugging
    print(updated_df.head())

if __name__ == "__main__":
    main()  # Run the main function if this script is executed
