import numpy as np
import pandas as pd


def total_tix_sold(list_of_dict):
    '''
    Args: 
        list of dictionaries from the ticket_types field in the data
    Returns:
        Sum of number tickets sold for that event
    '''
    sold = []
    for dicts in list_of_dict:
        sold.append(dicts['quantity_sold'])
    if np.sum(sold):
        return np.sum(sold)
    else:
        return 0


def sketchy_names(name):
    '''
    Args: 
        name (str): Name of event
    Returns:
        (int): 1 if the name is all upercase, lowercase, none, 0 if none of the above
    '''
    if name == name.upper():
        return 1
    elif name == name.lower():
        return 1
    elif name == "":
        return 1
    else:
        return 0


def has_zero(list_of_dict):
    '''
    Args: 
        list of dictionaries from ticket_types column
    Returns:
        (int): 1 or 0 depending whether there are any tickets available
    '''
    availability = []
    for d in list_of_dict:
        availability.append(d['cost'])
    return 1 * (0 in availability)


def max_price(list_of_dict):
    '''
    Args: 
        list of dictionaries from ticket_types column
    Returns:
        0 or a max value for the price of the tickets
    '''
    availability = []
    for d in list_of_dict:
        availability.append(d['cost'])

    if availability:
        return max(availability)
    else:
        return 0


def get_X(df):
    '''
    Creates fraud column and engineered features.
    Args: 
        Data in a pandas dataframe
    Returns: 
        Cleaned data ready for prediction
    '''

    bad_emails = ['yahoo.com', 'hotmail.com', 'ymail.com', 'aol.com',
                  'lidf.co.uk', 'live.com', 'live.fr', 'yahoo.co.uk', 'rocketmail.com']
    df['total_sold'] = df['ticket_types'].apply(total_tix_sold)
    df['has_fbook'] = 1 * (df['org_facebook'] > 0)
    df['has_twitter'] = 1 * (df['org_twitter'] > 0)
    df['sketchy_email'] = df['email_domain'].apply(
        lambda x: 1 * (x in bad_emails))
    df['sketchy_name'] = df['name'].apply(sketchy_names)
    df['has_prev_payouts'] = df['previous_payouts'].apply(
        lambda x: 1 * (len(x) > 0))
    df['has_payout_type'] = df['payout_type'].apply(lambda x: 1 * (x != ""))
    df['less_than_one_sold'] = df['total_sold'] < 1
    df['has_free_tix'] = df['ticket_types'].apply(has_zero)
    df['max_price_below_80'] = 1 * (df['ticket_types'].apply(max_price) < 80)

    cols_to_keep = ['less_than_one_sold', 'has_fbook', 'has_twitter', 'sketchy_email', 'sketchy_name',
                    'has_prev_payouts', 'has_payout_type', 'has_logo', 'has_analytics', 'has_free_tix', 'max_price_below_80']

    X = df[cols_to_keep]
    return X
