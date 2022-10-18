import pandas as pd

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

# complete_covid_data = pd.read_csv('complete_covid_data.csv')
# us_covid_data = complete_covid_data.loc[complete_covid_data['iso_code'] == 'USA']
# us_covid_data.to_csv('./usa_covid_data.csv')


class DataPreprocessing:
    """This class preprocesses the data for the country and creates the state-wise data."""

    def __init__(self):
        """This method reads all the data files."""
        self.us_testing = pd.read_csv('./data/us_data/us_testing.csv')
        self.us_hospitalizations = pd.read_csv('./data/us_data/us_hospitalizations.csv')
        self.us_vaccinations = pd.read_csv('./data/us_data/us_vaccinations.csv')
        self.state_mapping = self.create_state_mapping()

    def create_population_dynamics(self):
        return

    def create_state_mapping(self):
        """This method maps the two letter state names to the full state names for consistency across datasets."""
        state_names = self.us_testing['state'].unique()
        mapping_dict = {'AS': 'American Samoa'}
        for state_name in state_names:
            mapping_dict[state_name] = self.us_testing.loc[self.us_testing['state'] == state_name]['state_name'].iloc[0]
        return mapping_dict

    def create_state_vaccination_data(self):
        """This method take the U.S. vaccination data and creates the state-wise vaccination data."""
        states = self.us_vaccinations['location'].unique()
        for state in states:
            if state in ['Bureau of Prisons', 'Dept of Defense', 'District of Columbia',
                         'Federated States of Micronesia', 'Guam', 'Indian Health Svc', 'Long Term Care',
                         'Republic of Palau', 'United States', 'Veterans Health', 'Virgin Islands']:
                # test = ['Hawaii', 'Ohio', 'Vermont', 'Virginia', 'West Virginia']
                continue

            state_vaccination = self.us_vaccinations.loc[self.us_vaccinations['location'] == state]
            state_vaccination = state_vaccination.reset_index(drop=True, inplace=False)

            imputation_columns = ['total_vaccinations', 'total_distributed',
                                  'people_vaccinated', 'people_fully_vaccinated']

            for column_name in imputation_columns:
                for i in range(len(state_vaccination)):
                    if str(state_vaccination[column_name][i]) == 'nan':
                        counter = 1
                        while str(state_vaccination[column_name][i + counter]) == 'nan':
                            counter += 1
                        for j in range(counter):
                            state_vaccination[column_name][i + j] = state_vaccination[column_name][i + j + counter]
                    else:
                        break

            for i in range(1, len(state_vaccination)):
                for column_name in imputation_columns:
                    if str(state_vaccination[column_name][i]) == 'nan':
                        counter = 1
                        while str(state_vaccination[column_name][i + counter]) == 'nan':
                            counter += 1
                        diff = state_vaccination[column_name][i + counter] - state_vaccination[column_name][i - 1]
                        for j in range(counter):
                            try:
                                state_vaccination[column_name][i + j] = int(
                                    state_vaccination[column_name][i + j - 1] + diff / (counter + 1))
                            except ValueError:
                                print(state, i, column_name, diff, counter, state_vaccination[column_name][i + j - 1])
                                break

            # Part 2
            imputation_columns = ['people_fully_vaccinated_per_hundred', 'total_vaccinations_per_hundred',
                                  'people_vaccinated_per_hundred', 'distributed_per_hundred']

            for column_name in imputation_columns:
                for i in range(len(state_vaccination)):
                    if str(state_vaccination[column_name][i]) == 'nan':
                        counter = 1
                        while str(state_vaccination[column_name][i + counter]) == 'nan':
                            counter += 1
                        for j in range(counter):
                            state_vaccination[column_name][i + j] = state_vaccination[column_name][i + j + counter]
                    else:
                        break

            for i in range(1, len(state_vaccination)):
                for column_name in imputation_columns:
                    if str(state_vaccination[column_name][i]) == 'nan':
                        counter = 1
                        while str(state_vaccination[column_name][i + counter]) == 'nan':
                            counter += 1
                        diff = state_vaccination[column_name][i + counter] - state_vaccination[column_name][i - 1]
                        for j in range(counter):
                            state_vaccination[column_name][i + j] = float(
                                state_vaccination[column_name][i + j - 1] + diff / (counter + 1))

            booster_columns = ['total_boosters', 'total_boosters_per_hundred']
            for column_name in booster_columns:
                for i in range(len(state_vaccination)):
                    if str(state_vaccination[column_name][i]) == 'nan':
                        state_vaccination[column_name][i] = 0
                    else:
                        break

            for i in range(len(state_vaccination)):
                for column_name in booster_columns:
                    if str(state_vaccination[column_name][i]) == 'nan':
                        counter = 1
                        while str(state_vaccination[column_name][i + counter]) == 'nan':
                            counter += 1
                        diff = state_vaccination[column_name][i + counter] - state_vaccination[column_name][i - 1]
                        for j in range(counter):
                            state_vaccination[column_name][i + j] = int(
                                state_vaccination[column_name][i + j - 1] + diff / (counter + 1))

            state_vaccination.to_csv(f'./data/state_vaccinations/{state}_vaccination.csv', index=False)

    def create_state_hospitalization_data(self):
        """This method takes the U.S. hospitalization data and creates the state-wise hospitalization data."""
        states = self.us_hospitalizations['state'].unique()

        for state in states:
            state_hospitalization = self.us_hospitalizations.loc[self.us_hospitalizations['state'] == state]
            state_hospitalization = state_hospitalization.sort_values(by='date')
            state = self.state_mapping[state]
            state_hospitalization.to_csv(f'./data/state_hospitalizations/{state}_hospitalization.csv', index=False)

    def create_state_testing_data(self):
        states = self.us_testing['state_name'].unique()
        for state in states:
            state_testing = self.us_testing.loc[self.us_testing['state_name'] == state]
            state_dataframe = pd.DataFrame(columns=['date', 'new_tests', 'total_tests', 'new_negative_tests',
                                                    'total_negative_tests', 'new_positive_tests',
                                                    'total_positive_tests', 'new_inconclusive_tests',
                                                    'total_inconclusive_tests'])
            state_dataframe['date'] = state_testing['date'].unique()

            new_tests, total_tests, new_negative, total_negative, new_positive, total_positive, new_inconclusive, \
                total_inconclusive = [], [], [], [], [], [], [], []
            previous_date = None
            for date in state_testing['date']:
                if date != previous_date:
                    new_tests.append(sum(state_testing.loc[state_testing['date'] == date]['new_results_reported']))
                    total_tests.append(sum(state_testing.loc[state_testing['date'] == date]['total_results_reported']))

                    temp = state_testing.loc[((state_testing['date'] == date) &
                                              (state_testing['overall_outcome'] == 'Negative'))]['new_results_reported']
                    new_negative.append(0 if len(temp) == 0 else int(temp))

                    temp = state_testing.loc[((state_testing['date'] == date) &
                                              (state_testing['overall_outcome'] == 'Positive'))]['new_results_reported']
                    new_positive.append(0 if len(temp) == 0 else int(temp))

                    temp = \
                        state_testing.loc[((state_testing['date'] == date) &
                                           (state_testing['overall_outcome'] == 'Negative'))]['total_results_reported']
                    total_negative.append(0 if len(temp) == 0 else int(temp))

                    temp = \
                        state_testing.loc[((state_testing['date'] == date) &
                                           (state_testing['overall_outcome'] == 'Positive'))]['total_results_reported']
                    total_positive.append(0 if len(temp) == 0 else int(temp))

                    temp = \
                        state_testing.loc[((state_testing['date'] == date) & (
                                state_testing['overall_outcome'] == 'Inconclusive'))]['new_results_reported']
                    new_inconclusive.append(0 if len(temp) == 0 else int(temp))

                    temp = \
                        state_testing.loc[((state_testing['date'] == date) & (
                                state_testing['overall_outcome'] == 'Inconclusive'))]['total_results_reported']
                    total_inconclusive.append(0 if len(temp) == 0 else int(temp))

                    previous_date = date

            state_dataframe['new_tests'] = new_tests
            state_dataframe['total_tests'] = total_tests
            state_dataframe['new_negative_tests'] = new_negative
            state_dataframe['new_positive_tests'] = new_positive
            state_dataframe['total_negative_tests'] = total_negative
            state_dataframe['total_positive_tests'] = total_positive
            state_dataframe['new_inconclusive_tests'] = new_inconclusive
            state_dataframe['total_inconclusive_tests'] = total_inconclusive

            state_dataframe.to_csv(f'./data/state_testing/{state}_testing.csv', index=False)

    @staticmethod
    def create_state_final_dataset():
        population_dynamics = pd.read_csv('./covid_ny.csv')
        vaccination_data = pd.read_csv('./New York_vaccination.csv')
        hospitalization_data = pd.read_csv('./New York_hospitalization.csv')
        testing_data = pd.read_csv('./New York_testing.csv')

        hospitalization_data['date'] = pd.to_datetime(hospitalization_data['date'])
        testing_data['date'] = pd.to_datetime(testing_data['date'])
        population_dynamics = population_dynamics.merge(vaccination_data, how='inner', on='date')
        population_dynamics['date'] = pd.to_datetime(population_dynamics['date'])
        population_dynamics = population_dynamics.merge(hospitalization_data, how='inner', on='date')
        population_dynamics = population_dynamics.merge(testing_data, how='inner', on='date')
        population_dynamics.to_csv('./final_new_york.csv', index=False)
        print(population_dynamics.shape)
        return

    @staticmethod
    def drop_columns():
        final_data = pd.read_csv('./New_York.csv')
        print(final_data.shape)
        final_data = final_data.drop(columns=['location', 'state', 'geocoded_state'])
        print(final_data.shape)
        final_data.to_csv('./New_York.csv', index=False)


dp = DataPreprocessing()
# dp.create_state_vaccination_data()
# dp.create_state_testing_data()
# dp.create_state_hospitalization_data()
# dp.create_state_final_dataset()
dp.drop_columns()
