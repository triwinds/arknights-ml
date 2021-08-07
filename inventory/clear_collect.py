import os
import shutil


def clear_collect_folder(collect_path, retain_manual_files=True):
    for path in os.listdir(collect_path):
        full_path = os.path.join(collect_path, path)
        if os.path.isdir(full_path):
            if retain_manual_files and path.startswith('@'):
                continue
            print(f'remove {full_path}')
            shutil.rmtree(full_path)


if __name__ == '__main__':
    clear_collect_folder('images/collect')
    # clear_collect_folder('images/collect', False)
