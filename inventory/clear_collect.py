import os
import shutil


retain_folders = {'4001'}


def clear_collect_folder(collect_path, retain_manual_files=True):
    for path in os.listdir(collect_path):
        if path in retain_folders:
            print(f'{path} is in retain_folders, skip.')
            continue
        full_path = os.path.join(collect_path, path)
        if os.path.isdir(full_path):
            if retain_manual_files and path.startswith('@'):
                continue
            print(f'remove {full_path}')
            shutil.rmtree(full_path)


if __name__ == '__main__':
    clear_collect_folder('images/collect')
    # clear_collect_folder('images/collect', False)
