import os
GOOGLE_API_KEY = "AIzaSyATvOSTGTLgG0TF5mIWU61LkPXq3pYYdT4"


def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or 'ID' in key:
            os.environ[key] = value