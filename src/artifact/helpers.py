# %%
import pickle
from pathlib import Path
from colorama import Fore, Style
from IPython.display import display

# %%
REGRESSION_PROFILE_PATH = (
    Path.cwd().parent / 'models' / 'selection' / 'learner_profiles.pkl'
)


class RegressionProfile:
    def __init__(self, load_path=None):
        self.error_dataframes = dict()
        if load_path:
            try:
                self.load(load_path)
            except FileNotFoundError:
                pass

    def __repr__(self):
        keys = self.error_dataframes.keys()
        if keys is not None:
            return f'RegressionProfile object with keys {list(keys)}'
        else:
            return 'Uninitialized RegressionProfile object'

    def add_results(self, name, error_dataframe):
        self.error_dataframes[name] = error_dataframe

    def save(self, save_path):
        with open(save_path, 'wb') as file:
            pickle.dump(self.error_dataframes, file, pickle.HIGHEST_PROTOCOL)

    def load(self, load_path):
        with open(load_path, 'rb') as file:
            self.error_dataframes = pickle.load(file)

    def summarize(self, name):
        try:
            df = self.error_dataframes[name]
        except KeyError:
            print(f'No error summary with key {name} was found.')
            return
        best_learners = df.idxmin()
        print(1 * '\n')
        print(Fore.YELLOW + name + '\n' + len(name) * '-')
        # print(len(name) * '-')
        print(Style.RESET_ALL)
        print('Best learners total by response:')
        display(best_learners.value_counts(), best_learners.sort_values())
        print('\n\nSorted by median RMS error (smallest to largest):')
        display(df.T.describe().T.sort_values(by=['50%']))
        print('\n\nRMS Errors:')
        display(df)
        print(2 * '\n')
