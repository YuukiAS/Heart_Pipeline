import os


def check_existing_file(files_to_check, existing_path):
    """
    Check if the files in files_to_check already exist in existing_path.
    If so, return True, otherwise return False.
    """
    files_in_dir = os.listdir(existing_path)
    files_to_check = [x for x in files_to_check if x not in files_in_dir]
    if len(files_to_check) == 0:
        return True
    return False
