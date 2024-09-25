import re

def preprocess_input(user_input):
    # Bersihkan input, contoh sederhana (dapat dikembangkan)
    cleaned_input = re.sub(r'[^\w\s]', '', user_input)
    return cleaned_input
