import alexlib.toolbox as tb
import alexlib

"""To get the repo to work seamlessly across different machines, 
either create symlinks at home directory that point to the actual paths or this code must be changed to reflect
the paths on a certain machine.

Run this file in elevation status, and configure the paths one by one.
"""

__version__ = "5.0.0"
version = f"\nDeepHead V {__version__}\n alexlib V {alexlib.__version__}\n"
g_drive_path = tb.P.home() / "my_g_drive"  # symlink to tb.P(r'G:\\')
my_data_path = tb.P.home() / "my_data"  # symlink to tb.P(r"D:\my_data")
crogl_path = tb.P.home() / "my_crogl"  # symlink to tb.P(r"C:\Program Files (x86)\MRIcroGL\MRIcroGL.exe")

g = g_drive_path
d = my_data_path
dh = tb.P.cwd().split(at="deephead")[0] / "deephead"


def configure_path(path, name=None, home=None):
    if path == '':
        print("No symlink was established.")
        return None  # quite the function.
    if home is None:
        home = tb.P.home()
    path = tb.P(path)
    if name is None:
        name = path[-1]
    if home.joinpath(name).exists():
        print(f"Symlink already exists ... skipping")
    else:
        home.joinpath(name).symlink_to(path, target_is_directory=True)
        print(f"Success! Now, {home.joinpath(name)} points to {path}")


def setup():
    home = input(f"Please enter your user home path (not system) (default value: {tb.P.home()}): ")
    if home == '':
        home = tb.P.home()
    else:
        home = tb.P(home)
    path = input(f"Creating the symlink {g_drive_path}, \nPlease enter G drive path: ")
    configure_path(path, name=g_drive_path.stem, home=home)
    path = input(f"Creating the symlink {my_data_path}, \nPlease enter D drive path: ")
    configure_path(path, name=my_data_path.stem, home=home)
    path = input(f"Creating the symlink {crogl_path}, \nPlease enter MRIcrogl path: ")
    configure_path(path, name=crogl_path.stem, home=home)


if __name__ == '__main__':
    setup()
