import os
import collections
import six
import pandas as pd
import re

def is_iterable(arg):
    return (
        isinstance(arg, collections.Iterable) 
        and not isinstance(arg, six.string_types)
    )


def get_images_paths(input_path):
    final_images_paths = []
    
    items = []
    temp_items = []

    if is_iterable(input_path):
        temp_items.extend(input_path)
    else:
        temp_items.append(input_path)
    
    while len(temp_items):
        items = temp_items
        temp_items = []
        
        for item in items:
            extension = item[-4:]
            if extension != '.jpg':
                # we assume that item is actually a directory
                new_items = os.listdir(item)
                for new_item in new_items:
                    temp_items.append(item + f'/{new_item}')
            else:
                final_images_paths.append(item)
    
    return final_images_paths

def get_dataset(photo_paths, file, sheet_name=None):
    paths = get_images_paths(photo_paths)
    df = pd.read_excel(file, sheet_name=sheet_name)
    df.drop(columns='projectCode', inplace=True)
    df['forGoogleMap'] = df['forGoogleMap'].str.replace(', ', '-')
    addresses = df['forGoogleMap'].str.extract(re.compile(r'(\d+) ([\w ]+)-([\w ]+)'))
    numbers = addresses[0]
    temp_lens = numbers.str.len()
    max_digits = temp_lens.max()
    if temp_lens.min() < 1:
        raise Exception()
    for i in range(1, max_digits):
        numbers[temp_lens == i] = (max_digits - i) * '0' + numbers[temp_lens == i]
    addresses[0] = numbers
    
    df['forGoogleMap'] = photo_paths + '/' + addresses[0] + '-' +\
        addresses[1] + '-' + addresses[2] + '.jpg'
    df['forGoogleMapAlt'] = photo_paths + '/' + addresses[0] + '-' +\
        addresses[1] + ', ' + addresses[2] + '.jpg'
    dataset1 = pd.DataFrame({'Paths': paths})
    dataset1 = dataset1.join(df.set_index('forGoogleMap'), on='Paths')
    dataset2 = pd.DataFrame({'Paths': paths})
    dataset2 = dataset2.join(df.set_index('forGoogleMapAlt'), on='Paths')
    dataset = dataset1.combine_first(dataset2)
    dataset = dataset[['Paths', 'front roof angle']]
    dataset = dataset[~dataset['front roof angle'].isna()]
    
    return dataset

if __name__ == '__main__':
    path = './photos'
    file = 'Prich jobs stats.xlsx'
    sheet_name = 'code-address-roofPitch'
    dataset = get_dataset(path, file, sheet_name)
    
    

